import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.nets.conv_net import ConvNet
from lop.nets.conv_net2 import ConvNet2
from lop.algos.convCBP import ConvCBP
from lop.algos.sdp import apply_sdp
from lop.algos.ema import EMAWrapper
from lop.optimizers import get_optimizer
from torch.nn.functional import softmax
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy as accuracy
from lop.metrics import compute_task_metrics, print_task_summary

train_images_per_class = 600
test_images_per_class = 100
images_per_class = train_images_per_class + test_images_per_class


def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = 'data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def save_data(data, data_file):
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)


class SecondOrderLearnerConv:
    """Learner that wraps second-order optimizers for ConvNet ImageNet experiments.
    Mirrors Backprop.learn() interface."""

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
        if self.optimizer_type == 'sassha':
            return self._learn_sassha(x, target)
        elif self.optimizer_type in ('adahessian', 'sophia', 'sophiah'):
            return self._learn_hessian(x, target)
        else:
            return self._learn_standard(x, target)

    def _learn_standard(self, x, target):
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        self.previous_features = features
        loss = self.loss_func(output, target)
        loss.backward()
        self.opt.step()
        return loss.detach(), output.detach()

    def _learn_hessian(self, x, target):
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        self.previous_features = features
        loss = self.loss_func(output, target)
        loss.backward(create_graph=True)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return loss.detach(), output.detach()

    def _learn_sassha(self, x, target):
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
        return loss.detach(), output.detach()

    def fade_optimizer_state(self):
        if self.optimizer_type in ('adahessian', 'sophia', 'sophiah', 'sassha'):
            for state in self.opt.state.values():
                if 'exp_avg' in state: state['exp_avg'].mul_(0.5)
                if 'exp_hessian_diag' in state: state['exp_hessian_diag'].mul_(0.5)
                if 'exp_hessian_diag_sq' in state: state['exp_hessian_diag_sq'].mul_(0.5)
        elif self.optimizer_type == 'kfac' and hasattr(self.opt, 'reset_stats'):
            self.opt.reset_stats()


def repeat_expr(params: {}):
    agent_type = params['agent']
    num_tasks = params['num_tasks']
    num_showings = params['num_showings']

    step_size = params['step_size']
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'contribution'
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev='cpu'
    num_classes = 10
    total_classes = 1000
    new_heads = True
    mini_batch_size = 100
    perturb_scale = 0
    momentum = 0
    net_type = 1
    if 'replacement_rate' in params.keys(): replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys(): decay_rate = params['decay_rate']
    if 'util_type' in params.keys(): util_type = params['util_type']
    if 'maturity_threshold' in params.keys():   maturity_threshold = params['maturity_threshold']
    if 'weight_decay' in params.keys(): weight_decay = params['weight_decay']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'num_classes' in params.keys():  num_classes = params['num_classes']
    if 'new_heads' in params.keys():    new_heads = params['new_heads']
    if 'mini_batch_size' in params.keys():  mini_batch_size = params['mini_batch_size']
    if 'perturb_scale' in params.keys():    perturb_scale = params['perturb_scale']
    if 'momentum' in params.keys(): momentum = params['momentum']
    if 'net_type' in params.keys(): net_type = params['net_type']
    num_epochs = num_showings

    classes_per_task = num_classes
    net = ConvNet()
    if net_type == 2:
        net = ConvNet2(replacement_rate=replacement_rate, maturity_threshold=maturity_threshold)
    if agent_type == 'linear':
        net = MyLinear( 
            input_size=3072, num_outputs=classes_per_task
        )

    if agent_type in ['bp', 'linear']:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            to_perturb=(perturb_scale != 0),
            perturb_scale=perturb_scale,
            device=dev,
            momentum=momentum,
        )
    elif agent_type == 'cbp':
        learner = ConvCBP(
            net=net,
            step_size=step_size,
            momentum=momentum,
            loss='nll',
            weight_decay=weight_decay,
            opt=opt,
            init='default',
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            util_type=util_type,
            device=dev,
            maturity_threshold=maturity_threshold,
        )
    elif agent_type == 'secondorder':
        optimizer_type = params.get('optimizer_type', 'adahessian')
        optimizer_params = params.get('optimizer_params', {})
        sdp_gamma = params.get('sdp_gamma', 0.0)
        use_ema = params.get('use_ema', False)
        ema_decay = params.get('ema_decay', 0.999)
        learner = SecondOrderLearnerConv(
            net=net,
            optimizer_type=optimizer_type,
            step_size=step_size,
            weight_decay=weight_decay,
            optimizer_params=optimizer_params,
            device=dev,
            to_perturb=(perturb_scale != 0),
            perturb_scale=perturb_scale,
        )
        ema = EMAWrapper(net, ema_decay) if use_ema else None
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    with open('class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[int([params['run_idx']][0])]
    num_class_repetitions_required = int(num_classes * num_tasks / total_classes) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    examples_per_epoch = train_images_per_class * classes_per_task

    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)

    # Metric history
    metrics_before = []
    metrics_after = []
    prev_ref_outputs = None

    x_train, x_test, y_train, y_test = None, None, None, None
    for task_idx in range(num_tasks):
        del x_train, x_test, y_train, y_test
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx*classes_per_task:(task_idx+1)*classes_per_task])
        x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
        if agent_type == 'linear':
            x_train, x_test = x_train.flatten(1), x_test.flatten(1)
        if use_gpu == 1:
            x_train, x_test, y_train, y_test = x_train.to(dev), x_test.to(dev), y_train.to(dev), y_test.to(dev)
        if new_heads:
            net.layers[-1].weight.data *= 0
            net.layers[-1].bias.data *= 0

        # ── BEFORE task: compute all metrics ──
        ref_batch = x_test[:64]
        _, feats_before = net.predict(x=ref_batch)
        before_metrics = compute_task_metrics(
            net, ref_batch, feats_before,
            prev_ref_outputs=prev_ref_outputs, loss_type='ce')

        # ── Train on task ──
        task_loss_sum = 0.0
        task_loss_count = 0
        for epoch_idx in tqdm(range(num_epochs)):
            example_order = np.random.permutation(train_images_per_class * classes_per_task)
            x_train = x_train[example_order]
            y_train = y_train[example_order]
            new_train_accuracies = torch.zeros((int(examples_per_epoch/mini_batch_size),), dtype=torch.float)
            epoch_iter = 0
            for start_idx in range(0, examples_per_epoch, mini_batch_size):
                batch_x = x_train[start_idx: start_idx+mini_batch_size]
                batch_y = y_train[start_idx: start_idx+mini_batch_size]

                loss, network_output = learner.learn(x=batch_x, target=batch_y)
                task_loss_sum += loss.item() if hasattr(loss, 'item') else float(loss)
                task_loss_count += 1

                # EMA update (only for secondorder agent)
                if agent_type == 'secondorder' and ema is not None:
                    ema.update(net)

                with torch.no_grad():
                    new_train_accuracies[epoch_iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
                    epoch_iter += 1

            with torch.no_grad():
                train_accuracies[task_idx][epoch_idx] = new_train_accuracies.mean()
                new_test_accuracies = torch.zeros((int(x_test.shape[0] / mini_batch_size),), dtype=torch.float)
                test_epoch_iter = 0
                for start_idx in range(0, x_test.shape[0], mini_batch_size):
                    test_batch_x = x_test[start_idx: start_idx + mini_batch_size]
                    test_batch_y = y_test[start_idx: start_idx + mini_batch_size]

                    network_output, _ = net.predict(x=test_batch_x)
                    new_test_accuracies[test_epoch_iter] = accuracy(softmax(network_output, dim=1), test_batch_y)
                    test_epoch_iter += 1

                test_accuracies[task_idx][epoch_idx] = new_test_accuracies.mean()

        # ── AFTER task: compute all metrics ──
        cur_output, feats_after = net.predict(x=ref_batch)
        after_metrics = compute_task_metrics(
            net, ref_batch, feats_after,
            prev_ref_outputs=prev_ref_outputs, loss_type='ce')
        prev_ref_outputs = cur_output.detach()

        metrics_before.append(before_metrics)
        metrics_after.append(after_metrics)

        task_loss = task_loss_sum / max(task_loss_count, 1)
        train_acc = train_accuracies[task_idx].mean().item()
        test_acc = test_accuracies[task_idx, -1].item()

        print_task_summary(task_idx, before_metrics, after_metrics,
                           task_loss, train_acc, test_acc)

        # SDP at task boundary (only for secondorder agent)
        if agent_type == 'secondorder' and sdp_gamma > 0:
            apply_sdp(net, sdp_gamma)
            learner.fade_optimizer_state()
            if ema is not None:
                ema.reset(net)

        if task_idx % save_after_every_n_tasks == 0:
            save_data(data={
                'train_accuracies': train_accuracies.cpu(),
                'test_accuracies': test_accuracies.cpu(),
                'metrics_before': metrics_before,
                'metrics_after': metrics_after,
            }, data_file=params['data_file'])
    
    save_data(data={
        'train_accuracies': train_accuracies.cpu(),
        'test_accuracies': test_accuracies.cpu(),
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
    }, data_file=params['data_file'])


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

    repeat_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
