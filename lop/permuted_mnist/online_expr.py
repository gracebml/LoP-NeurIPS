import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.algos.sdp import apply_sdp
from lop.algos.ema import EMAWrapper
from lop.optimizers import get_optimizer
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy
from lop.metrics import compute_task_metrics, print_task_summary


def _get_features(net, x):
    """Get model output and features for metric computation."""
    output, features = net.predict(x)
    return output, features


class SecondOrderLearner:
    """Learner that wraps second-order optimizers (AdaHessian, SophiaH, KFAC, SASSHA)
    for the MNIST experiment. Mirrors the interface of Backprop.learn()."""

    def __init__(self, net, optimizer_type, step_size, weight_decay=0.0,
                 optimizer_params=None, device='cpu', to_perturb=False, perturb_scale=0.1):
        import torch.nn.functional as F
        self.net = net
        self.device = device
        self.optimizer_type = optimizer_type.lower()
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.previous_features = None
        self.loss_func = F.cross_entropy

        kwargs = dict(lr=step_size, weight_decay=weight_decay)
        if optimizer_params:
            kwargs.update(optimizer_params)
        self.opt = get_optimizer(optimizer_type, net, **kwargs)

    def learn(self, x, target):
        """One training step. Returns (loss, output)."""
        if self.optimizer_type == 'sassha':
            return self._learn_sassha(x, target)
        elif self.optimizer_type in ('adahessian', 'sophia', 'sophiah'):
            return self._learn_hessian(x, target)
        else:
            return self._learn_standard(x, target)

    def _learn_standard(self, x, target):
        """Standard forward-backward for SGD/Adam/KFAC."""
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        self.previous_features = features
        loss = self.loss_func(output, target)
        loss.backward()
        self.opt.step()
        if self.to_perturb:
            self._perturb()
        return loss.detach(), output.detach()

    def _learn_hessian(self, x, target):
        """AdaHessian/SophiaH: need create_graph=True."""
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        self.previous_features = features
        loss = self.loss_func(output, target)
        loss.backward(create_graph=True)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if self.to_perturb:
            self._perturb()
        return loss.detach(), output.detach()

    def _learn_sassha(self, x, target):
        """SASSHA two-pass protocol."""
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        self.previous_features = features
        loss = self.loss_func(output, target)
        loss.backward()
        self.opt.perturb_weights(zero_grad=True)
        output_pert, _ = self.net.predict(x=x)
        loss_pert = self.loss_func(output_pert, target)
        loss_pert.backward(create_graph=True)
        self.opt.unperturb()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if self.to_perturb:
            self._perturb()
        return loss.detach(), output.detach()

    def _perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias += \
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight += \
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)

    def fade_optimizer_state(self):
        """Fade optimizer state after SDP."""
        if self.optimizer_type in ('adahessian', 'sophia', 'sophiah', 'sassha'):
            for state in self.opt.state.values():
                if 'exp_avg' in state: state['exp_avg'].mul_(0.5)
                if 'exp_hessian_diag' in state: state['exp_hessian_diag'].mul_(0.5)
                if 'exp_hessian_diag_sq' in state: state['exp_hessian_diag_sq'].mul_(0.5)
        elif self.optimizer_type == 'kfac' and hasattr(self.opt, 'reset_stats'):
            self.opt.reset_stats()


def online_expr(params: {}):
    agent_type = params['agent']
    num_tasks = 200
    if 'num_tasks' in params.keys():
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"]/params["change_after"])

    step_size = params['step_size']
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev = 'cpu'
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 1
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'

    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'num_features' in params.keys():
        num_features = params['num_features']
    if 'change_after' in params.keys():
        change_after = params['change_after']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in params.keys():
        to_perturb = params['to_perturb']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']
    if 'num_hidden_layers' in params.keys():
        num_hidden_layers = params['num_hidden_layers']
    if 'mini_batch_size' in params.keys():
        mini_batch_size = params['mini_batch_size']
    if 'replacement_rate' in params.keys():
        replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys():
        decay_rate = params['decay_rate']
    if 'maturity_threshold' in params.keys():
        maturity_threshold = params['mt']
    if 'util_type' in params.keys():
        util_type = params['util_type']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    num_hidden_layers = num_hidden_layers
    net = DeepFFNN(input_size=input_size, num_features=num_features, num_outputs=classes_per_task,
                   num_hidden_layers=num_hidden_layers)

    if agent_type == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []

    if agent_type in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
        )
    elif agent_type in ['secondorder']:
        optimizer_type = params.get('optimizer_type', 'adahessian')
        optimizer_params = params.get('optimizer_params', {})
        sdp_gamma = params.get('sdp_gamma', 0.0)
        use_ema = params.get('use_ema', False)
        ema_decay = params.get('ema_decay', 0.999)
        learner = SecondOrderLearner(
            net=net,
            optimizer_type=optimizer_type,
            step_size=step_size,
            weight_decay=weight_decay,
            optimizer_params=optimizer_params,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
        ema = EMAWrapper(net, ema_decay) if use_ema else None
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples/mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    accuracies = torch.zeros(total_iters, dtype=torch.float)
    weight_mag_sum = torch.zeros((total_iters, num_hidden_layers+1), dtype=torch.float)

    # Metric history storage
    num_checkpoints = int(total_examples / 60000)
    metrics_before = []
    metrics_after = []
    prev_ref_outputs = None

    iter = 0
    with open('data/mnist/mnist_', 'rb+') as f:
        x, y, x_test_all, y_test_all = pickle.load(f)
        if use_gpu == 1:
            x = x.to(dev)
            y = y.to(dev)
            x_test_all = x_test_all.to(dev)
            y_test_all = y_test_all.to(dev)

    for task_idx in (range(num_tasks)):
        new_iter_start = iter
        pixel_permutation = np.random.permutation(input_size)
        x = x[:, pixel_permutation]
        x_test_task = x_test_all[:, pixel_permutation]
        data_permutation = np.random.permutation(examples_per_task)
        x, y = x[data_permutation], y[data_permutation]

        # ── BEFORE task: compute all metrics ──
        before_metrics = None
        if agent_type != 'linear':
            ref_batch = x[:64]
            _, features = net.predict(ref_batch)
            before_metrics = compute_task_metrics(
                net, ref_batch, features,
                prev_ref_outputs=prev_ref_outputs, loss_type='ce')

        # ── Train on task ──
        task_loss_sum = 0.0
        task_loss_count = 0
        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
            start_idx = start_idx % examples_per_task
            batch_x = x[start_idx: start_idx+mini_batch_size]
            batch_y = y[start_idx: start_idx+mini_batch_size]

            loss, network_output = learner.learn(x=batch_x, target=batch_y)
            task_loss_sum += loss.item() if hasattr(loss, 'item') else float(loss)
            task_loss_count += 1

            # EMA update (only for secondorder agent)
            if agent_type == 'secondorder' and ema is not None:
                ema.update(net)

            if to_log and agent_type != 'linear':
                for idx, layer_idx in enumerate(learner.net.layers_to_log):
                    weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
            with torch.no_grad():
                accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
            iter += 1

        # ── Compute train_acc, test_acc, loss ──
        train_acc = accuracies[new_iter_start:iter].mean().item()
        task_loss = task_loss_sum / max(task_loss_count, 1)

        # Test accuracy on permuted test set
        with torch.no_grad():
            test_out, _ = net.predict(x_test_task)
            test_acc = accuracy(softmax(test_out, dim=1), y_test_all).item()

        # ── AFTER task: compute all metrics ──
        after_metrics = None
        if agent_type != 'linear':
            ref_batch = x[:64]
            cur_output, features = net.predict(ref_batch)
            after_metrics = compute_task_metrics(
                net, ref_batch, features,
                prev_ref_outputs=prev_ref_outputs, loss_type='ce')
            prev_ref_outputs = cur_output.detach()

            metrics_before.append(before_metrics)
            metrics_after.append(after_metrics)

            # ── Print full summary ──
            print_task_summary(task_idx, before_metrics, after_metrics,
                               task_loss, train_acc, test_acc)

            # SDP at task boundary (only for secondorder agent)
            if agent_type == 'secondorder' and sdp_gamma > 0:
                apply_sdp(net, sdp_gamma)
                learner.fade_optimizer_state()
                if ema is not None:
                    ema.reset(net)
        else:
            print(f'[Task {task_idx}] loss: {task_loss:.4f}, '
                  f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')

        if task_idx % save_after_every_n_tasks == 0:
            data = {
                'accuracies': accuracies.cpu(),
                'weight_mag_sum': weight_mag_sum.cpu(),
                'metrics_before': metrics_before,
                'metrics_after': metrics_after,
            }
            save_data(file=params['data_file'], data=data)

    data = {
        'accuracies': accuracies.cpu(),
        'weight_mag_sum': weight_mag_sum.cpu(),
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
    }
    save_data(file=params['data_file'], data=data)


def save_data(file, data):
    with open(file, 'wb+') as f:
        pickle.dump(data, f)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    online_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
