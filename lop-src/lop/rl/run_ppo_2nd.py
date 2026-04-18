"""
PPO with Second-Order Optimizers ± SDP on Continual RL (Ant-v4).

Based on notebooks/rl/rl-secondorder-sdp.py.
Uses existing lop/ modules for optimizers, SDP, and metrics.

Usage:
    python run_ppo_2nd.py -c cfg/ant/sassha_sdp.yml -s 1
"""
import os
import sys
import yaml
import time
import pickle
import argparse
import numpy as np

import gym
import torch
import torch.nn as nn
from tqdm import tqdm

from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.optimizers import get_optimizer
from lop.algos.sdp import apply_sdp
from lop.utils.miscellaneous import compute_matrix_rank_summaries


# ══════════════════════════════════════════════════════════════════
#  Metrics (adapted from notebook §2)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_dormant_units_rl(pol, vf, threshold=0.01):
    """Compute fraction of dormant neurons from stored activations."""
    total_units, dormant_units = 0, 0
    for net in [pol, vf]:
        if hasattr(net, 'activations') and net.activations:
            for key, feat in net.activations.items():
                if feat is not None and feat.dim() >= 2:
                    activity = (feat != 0).float().mean(dim=0)
                    dormant = (activity < threshold).sum().item()
                    dormant_units += dormant
                    total_units += activity.numel()
    return dormant_units / total_units if total_units > 0 else 0.0


@torch.no_grad()
def compute_stable_rank_from_features(feature_activity):
    """Compute stable rank from feature activations (99% variance)."""
    if feature_activity is None or feature_activity.numel() == 0:
        return 1.0
    _, _, _, stable_rank = compute_matrix_rank_summaries(
        m=feature_activity, prop=0.99, use_scipy=True)
    return stable_rank.item() if torch.is_tensor(stable_rank) else float(stable_rank)


@torch.no_grad()
def compute_weight_magnitude(pol, vf):
    """Compute average weight magnitude across both networks."""
    total, n = 0.0, 0
    for net in [pol, vf]:
        for name, p in net.named_parameters():
            if 'weight' in name:
                total += p.abs().mean().item()
                n += 1
    return total / n if n else 0.0


def apply_sdp_rl(pol, vf, gamma):
    """Apply SDP to both policy and value networks (skip output layers)."""
    cond_numbers = []
    for net_name, net in [('pol', pol), ('vf', vf)]:
        if hasattr(net, 'mean_net'):
            seq = net.mean_net
        elif hasattr(net, 'v_net'):
            seq = net.v_net
        else:
            continue
        modules = [m for m in seq.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for i, module in enumerate(modules):
                if i == len(modules) - 1:
                    continue  # skip output layer
                W = module.weight.data
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except Exception:
                    continue
                if S.numel() == 0 or S[0] < 1e-12:
                    continue
                cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
                s_mean = S.mean().clamp(min=1e-12)
                S_new = (s_mean ** gamma) * (S ** (1.0 - gamma))
                W_new = U @ torch.diag(S_new) @ Vh
                module.weight.data.copy_(W_new)
    return cond_numbers


# ══════════════════════════════════════════════════════════════════
#  Optimizer builder for RL (joint policy+value params)
# ══════════════════════════════════════════════════════════════════

def build_optimizer_rl(config, pol, vf):
    """Build optimizer from config for RL (joint policy+value)."""
    opt_type = config['optimizer']
    all_params = list(pol.parameters()) + list(vf.parameters())

    if opt_type == 'adam':
        return torch.optim.Adam(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.99, 0.99)),
            weight_decay=config.get('weight_decay', 0),
            eps=config.get('eps', 1e-8))

    elif opt_type == 'adahessian':
        from lop.optimizers.adahessian import Adahessian
        return Adahessian(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 0.5),
            lazy_hessian=config.get('lazy_hessian', 1),
            n_samples=config.get('n_samples', 1),
            seed=config.get('seed', 42))

    elif opt_type in ('sophia', 'sophiah'):
        from lop.optimizers.sophiaH import SophiaH
        return SophiaH(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.965, 0.99)),
            weight_decay=config.get('weight_decay', 0),
            eps=config.get('eps', 1e-4),
            clip_threshold=config.get('clip_threshold', 1.0),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1),
            seed=config.get('seed', 42))

    elif opt_type == 'sassha':
        from lop.optimizers.sassha import SASSHA
        return SASSHA(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0),
            rho=config.get('rho', 0.05),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 0.5),
            seed=config.get('seed', 42))

    elif opt_type == 'kfac':
        from lop.optimizers.kfac_ngd import KFACOptimizer
        # K-FAC needs a single nn.Module — wrap pol+vf
        class _CombinedNet(nn.Module):
            def __init__(self, pol, vf):
                super().__init__()
                self.pol = pol
                self.vf = vf
        combined = _CombinedNet(pol, vf)
        return KFACOptimizer(
            combined, lr=config['lr'],
            damping=config.get('damping', 1e-3),
            weight_decay=config.get('weight_decay', 0),
            T_inv=config.get('T_inv', 50),
            alpha=config.get('alpha', 0.95),
            max_grad_norm=config.get('max_grad_norm', 1.0))
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


# ══════════════════════════════════════════════════════════════════
#  PPO learning step (from notebook §5, without ASAM/Shampoo)
# ══════════════════════════════════════════════════════════════════

def _needs_create_graph(config):
    return config['optimizer'] in ('adahessian', 'sophia', 'sophiah', 'sassha')


def ppo_learn(pol, vf, opt, buf, config, grad_clip=1.0):
    """PPO learning step with second-order optimizer dispatch."""
    g = config.get('g', 0.99)
    lm = config.get('lm', 0.95)
    bs = config.get('bs', 2048)
    n_itrs = config.get('n_itrs', 10)
    n_slices = config.get('n_slices', 16)
    clip_eps = config.get('clip_eps', 0.2)
    opt_type = config['optimizer']
    create_graph = _needs_create_graph(config)

    os_t, acts, rs, op, logpbs, _, dones = buf.get(pol.dist_stack)

    with torch.no_grad():
        pre_vals = vf.value(torch.cat((os_t, op)))

    # GAE
    vals = pre_vals.squeeze()
    rs_sq = rs.squeeze()
    dones_sq = dones.squeeze()
    advs = torch.zeros(len(rs_sq) + 1, device=rs_sq.device)
    for t in reversed(range(len(rs_sq))):
        delta = rs_sq[t] + (1 - dones_sq[t]) * g * vals[t + 1] - vals[t]
        advs[t] = delta + (1 - dones_sq[t]) * g * lm * advs[t + 1]
    v_rets = advs[:-1] + vals[:-1]
    advs = advs[:-1].view(-1)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    v_rets = v_rets.view(-1).detach()
    advs = advs.detach()
    logpbs = logpbs.view(-1).detach()

    inds = np.arange(os_t.shape[0])
    mini_bs = bs // n_slices
    p_loss_val, v_loss_val = 0.0, 0.0
    all_params = list(pol.parameters()) + list(vf.parameters())

    for _ in range(n_itrs):
        np.random.shuffle(inds)
        for start in range(0, len(os_t), mini_bs):
            ind = inds[start:start + mini_bs]
            advs_i = advs[ind]
            v_rets_i = v_rets[ind]
            logpbs_i = logpbs[ind]

            if opt_type == 'sassha':
                # Two-pass: forward → backward → perturb → forward → backward(create_graph) → step
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss = torch.max(-(grad_sub * advs_i),
                                   -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)).mean()
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()

                if any(p.grad is not None and torch.isnan(p.grad).any() for p in all_params):
                    opt.zero_grad(set_to_none=True)
                    continue

                opt.perturb_weights(zero_grad=True)
                logpts2, _ = pol.logp_dist(os_t[ind], acts[ind])
                logpts2 = logpts2.view(-1)
                grad_sub2 = (logpts2 - logpbs_i).exp()
                p_loss2 = torch.max(-(grad_sub2 * advs_i),
                                    -(torch.clamp(grad_sub2, 1 - clip_eps, 1 + clip_eps) * advs_i)).mean()
                v_vals2 = vf.value(os_t[ind]).view(-1)
                v_loss2 = (v_rets_i - v_vals2).pow(2).mean()
                (p_loss2 + v_loss2).backward(create_graph=True)
                opt.unperturb()
                opt.step()
                opt.zero_grad(set_to_none=True)
                p_loss_val, v_loss_val = p_loss.item(), v_loss.item()

            elif opt_type in ('adahessian', 'sophia', 'sophiah'):
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss = torch.max(-(grad_sub * advs_i),
                                   -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)).mean()
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward(create_graph=True)
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                p_loss_val, v_loss_val = p_loss.item(), v_loss.item()

            elif opt_type == 'kfac':
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss = torch.max(-(grad_sub * advs_i),
                                   -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)).mean()
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                p_loss_val, v_loss_val = p_loss.item(), v_loss.item()

            else:
                # Adam or any standard optimizer
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss = torch.max(-(grad_sub * advs_i),
                                   -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)).mean()
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                p_loss_val, v_loss_val = p_loss.item(), v_loss.item()

    return {'p_loss': p_loss_val, 'v_loss': v_loss_val}


# ══════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════

def fade_optimizer_state(opt, opt_type):
    """Fade Hessian/momentum state after SDP to prevent phase mismatch."""
    if opt_type in ('adahessian', 'sophia', 'sophiah', 'sassha'):
        for state in opt.state.values():
            if 'exp_avg' in state: state['exp_avg'].mul_(0.5)
            if 'exp_hessian_diag' in state: state['exp_hessian_diag'].mul_(0.5)
            if 'exp_hessian_diag_sq' in state: state['exp_hessian_diag_sq'].mul_(0.5)
    elif opt_type == 'kfac' and hasattr(opt, 'reset_stats'):
        opt.reset_stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='Path to YAML config file')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-d', '--device', type=str, default='')
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed
    cfg['seed'] = seed

    # ── Defaults ──
    cfg.setdefault('env_name', 'Ant-v4')
    cfg.setdefault('n_steps', int(10e6))
    cfg.setdefault('h_dim', [256, 256])
    cfg.setdefault('act_type', 'ReLU')
    cfg.setdefault('init', 'lecun')
    cfg.setdefault('bs', 2048)
    cfg.setdefault('n_itrs', 10)
    cfg.setdefault('n_slices', 16)
    cfg.setdefault('clip_eps', 0.2)
    cfg.setdefault('g', 0.99)
    cfg.setdefault('lm', 0.95)
    cfg.setdefault('grad_clip', 1.0)
    cfg.setdefault('sdp_gamma', 0.0)
    cfg.setdefault('sdp_interval', 100000)
    cfg.setdefault('log_interval', 1000)
    cfg.setdefault('save_interval', 500000)
    cfg.setdefault('time_limit_hours', 11.5)

    n_steps = int(float(cfg['n_steps']))
    opt_type = cfg['optimizer']
    sdp_gamma = cfg['sdp_gamma']
    sdp_interval = cfg['sdp_interval']

    # ── Paths ──
    results_dir = cfg.get('dir', f'results/rl_2nd/{opt_type}_sdp{sdp_gamma}/')
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'{seed}.pkl')
    ckpt_path = os.path.join(results_dir, f'{seed}.pth')

    print(f"{'='*70}")
    print(f"  PPO + {opt_type} | SDP γ={sdp_gamma} | {cfg['env_name']} | seed={seed}")
    print(f"  {n_steps:,} steps | bs={cfg['bs']} | h_dim={cfg['h_dim']}")
    print(f"{'='*70}")

    # ── Seed ──
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Env ──
    env = gym.make(cfg['env_name'])
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # ── Networks ──
    h_dim = cfg['h_dim']
    pol = MLPPolicy(o_dim, a_dim, act_type=cfg['act_type'],
                    h_dim=h_dim, device=device, init=cfg['init'])
    vf = MLPVF(o_dim, act_type=cfg['act_type'],
               h_dim=h_dim, device=device, init=cfg['init'])

    # ── Optimizer ──
    opt = build_optimizer_rl(cfg, pol, vf)

    # ── Buffer ──
    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)

    # ── Results tracking ──
    R = {
        'episodic_returns': [], 'termination_steps': [],
        'dormant_units': [], 'weight_magnitude': [],
        'stable_rank': [], 'p_loss': [], 'v_loss': [],
        'sdp_cond': [],
    }

    # ── Feature tracking for stable rank ──
    short_term_features = torch.zeros(1000, h_dim[-1], device=device)
    feature_idx = 0

    # ── Checkpoint resume ──
    start_step = 0
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pol.load_state_dict(ckpt['pol'])
        vf.load_state_dict(ckpt['vf'])
        try:
            opt.load_state_dict(ckpt['opt'])
        except Exception as e:
            print(f"  Warning: could not restore optimizer: {e}")
        R = ckpt.get('results', R)
        start_step = ckpt['step'] + 1
        feature_idx = ckpt.get('feature_idx', 0)
        print(f"  ✓ Resumed from step {start_step:,}")
        del ckpt
        torch.cuda.empty_cache()

    # ── Training loop ──
    wall_start = time.time()
    time_limit = cfg['time_limit_hours'] * 3600
    o = env.reset()
    if isinstance(o, tuple):
        o = o[0]  # gymnasium API
    episode_return = 0
    episode_count = len(R['episodic_returns'])

    pbar = tqdm(range(start_step, n_steps), desc=f"{opt_type}+sdp{sdp_gamma}",
                initial=start_step, total=n_steps)
    for step in pbar:
        if time.time() - wall_start > time_limit:
            print(f"\n  Time limit reached. Saving checkpoint.")
            break

        # ── Get action ──
        with torch.no_grad():
            a, logp, dist = pol.action(
                torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0),
                to_log_features=True)
            # Track last hidden layer features
            if hasattr(pol, 'activations') and pol.activations:
                features = list(pol.activations.values())
                for fi in reversed(features):
                    if fi is not None and fi.dim() >= 2 and fi.shape[-1] == h_dim[-1]:
                        short_term_features[feature_idx % 1000] = fi[0]
                        feature_idx += 1
                        break

        a_np = a[0].cpu().numpy()

        # ── Step env ──
        result = env.step(a_np)
        if len(result) == 5:
            op, r, terminated, truncated, info = result
            done = terminated or truncated
        else:
            op, r, done, info = result

        episode_return += r

        # ── Store transition ──
        buf.store(o, a_np, r, op, logp.cpu().numpy(),
                  pol.dist_to(dist, to_device='cpu'), float(done))
        o = op

        # ── Episode ended ──
        if done:
            R['episodic_returns'].append(episode_return)
            R['termination_steps'].append(step)
            episode_count += 1
            episode_return = 0
            o = env.reset()
            if isinstance(o, tuple):
                o = o[0]

        # ── PPO update when buffer full ──
        if len(buf.o_buf) >= cfg['bs']:
            learn_logs = ppo_learn(pol, vf, opt, buf, cfg,
                                   grad_clip=cfg['grad_clip'])
            buf.clear()
            R['p_loss'].append(learn_logs['p_loss'])
            R['v_loss'].append(learn_logs['v_loss'])
            R['weight_magnitude'].append(compute_weight_magnitude(pol, vf))
            R['dormant_units'].append(compute_dormant_units_rl(pol, vf))

        # ── Periodic SDP ──
        if step > 0 and step % sdp_interval == 0 and sdp_gamma > 0:
            cond_nums = apply_sdp_rl(pol, vf, sdp_gamma)
            avg_cond = sum(cond_nums) / max(len(cond_nums), 1) if cond_nums else 0.0
            R['sdp_cond'].append(avg_cond)
            fade_optimizer_state(opt, opt_type)
            print(f"\n  SDP at step {step:,}: avg cond={avg_cond:.1f}")

        # ── Stable rank every 10K steps ──
        if step > 0 and step % 10000 == 0:
            valid = min(feature_idx, 1000)
            if valid > 10:
                sr = compute_stable_rank_from_features(
                    short_term_features[:valid].cpu())
                R['stable_rank'].append(sr)

        # ── Progress bar ──
        if step > 0 and step % cfg['log_interval'] == 0:
            recent = R['episodic_returns'][-10:] if R['episodic_returns'] else [0]
            pbar.set_postfix({
                'Eps': episode_count,
                'Ret': f'{np.mean(recent):.0f}',
                'Dorm': f'{(R["dormant_units"][-1]*100 if R["dormant_units"] else 0):.1f}%',
            })

        # ── Checkpoint ──
        if step > 0 and step % cfg['save_interval'] == 0:
            torch.save({
                'step': step, 'pol': pol.state_dict(), 'vf': vf.state_dict(),
                'opt': opt.state_dict(), 'results': R, 'feature_idx': feature_idx,
            }, ckpt_path)
            print(f"\n  Checkpoint at step {step:,}")

    # ── Final save ──
    env.close()
    torch.save({
        'step': step, 'pol': pol.state_dict(), 'vf': vf.state_dict(),
        'opt': opt.state_dict(), 'results': R, 'feature_idx': feature_idx,
    }, ckpt_path)
    with open(log_path, 'wb') as f:
        pickle.dump(R, f)
    print(f"\n✓ Done: {episode_count} episodes, "
          f"final avg return = {np.mean(R['episodic_returns'][-100:]):.1f}")


if __name__ == "__main__":
    main()
