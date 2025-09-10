import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import numpy as np

model_names = ['resnet101', 'resnet50', 'resnet34', 'vgg16', 'vgg16_bn', 'resnet18', 'vgg11', 'vgg11_bn', 'mobilenet', 'resnet50_nobn']
eps_zero = 1e-15


def extract_layers(model, layers={}, pre_name='', get_conv=True, get_bn=False, get_gate=False):
    """
        Get all the model layers and the names of the layers
    Returns:
        layers: dict, the key is layer name and the value is the corresponding model layer
    """
    for name, layer in model.named_children():
        new_name = '.'.join([pre_name,name]) if pre_name != '' else name
        if len(list(layer.named_children())) > 0:
            extract_layers(layer, layers, new_name, get_conv, get_bn, get_gate)
        else:
            get_layer = False
            if get_conv and is_conv(layer):
                get_layer = True
            elif get_bn and is_bn(layer):
                get_layer = True
            elif get_gate and 'gate' in new_name:
                get_layer = True
            if get_layer:
                layers[new_name] = layer
    return layers


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_conv(layer):
    conv_type = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    isConv = False
    for ctp in conv_type:
        if isinstance(layer, ctp):
            isConv = True
            break
    return isConv


def is_bn(layer):
    bn_type = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    isbn = False
    for ctp in bn_type:
        if isinstance(layer, ctp):
            isbn = True
            break
    return isbn


def count_non_zero(input_tensor):
    """
        count the number of non-zero parameters in the input tensor
    """
    cmp = torch.ge(torch.abs(input_tensor), eps_zero)
    count = torch.sum(cmp.type(torch.float32))
    return int(count)


def count_non_zero_neurons(layers, layer_neuron_num=None, neuron_norm=None, verbose=False):
    """
        count the total number of (conv) neurons
        get the neuron norm if get_avg_norm is set to True
    Returns:
        total_num: int
    """
    total_num = 0
    for name, layer in layers.items():
        if is_conv(layer):
            norm = layer.weight.data.view(layer.weight.data.size(0), -1).norm(dim=1)
            cur_num = count_non_zero(norm)
            total_num += cur_num
            if layer_neuron_num is not None:
                layer_neuron_num[name].append(cur_num)
            if neuron_norm is not None:
                avg_norm = float(norm.sum().item() / cur_num)
                print('layer {} AvgNorm: {}'.format(name, avg_norm))
                neuron_norm[name].append(norm.detach().cpu().numpy().squeeze())
            if verbose:
                print('{} has {} neurons.'.format(name, cur_num))
    return total_num

def count_non_zero_neurons_softmask(layers, layer_neuron_num=None, neuron_norm=None, verbose=False):
    """
        count the total number of (conv) neurons
        get the neuron norm if get_avg_norm is set to True
    Returns:
        total_num: int
    """
    total_num = 0
    for name, layer in layers.items():
        if is_conv(layer):
            norm = layer.weight_orig.data.view(layer.weight_orig.data.size(0), -1).norm(dim=1)
            cur_num = count_non_zero(norm)
            total_num += cur_num
            if layer_neuron_num is not None:
                layer_neuron_num[name].append(cur_num)
            if neuron_norm is not None:
                avg_norm = float(norm.sum().item() / cur_num)
                print('layer {} AvgNorm: {}'.format(name, avg_norm))
                neuron_norm[name].append(norm.detach().cpu().numpy().squeeze())
            if verbose:
                print('{} has {} neurons.'.format(name, cur_num))
    return total_num

def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser(description='Script for testing ideas of pruning in pytorch')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--enable-bias', type=bool, default=False,
                        help='Whether to enable the bias term in the convolution layers')
    parser.add_argument('--pruned', type=bool, default=False,
                        help='Whether to use the pruned structure as the starting point.')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate layers in the network')
    parser.add_argument('--dataset-name', type=str, default='ImageNet',
                        help='The name of the dataset')
    parser.add_argument('--data-root', type=str, default='/mnt/data/',
                        help='The root directory of the dataset')
    parser.add_argument('--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The number of examples in one training batch')
    parser.add_argument('--optimizer-batch-size', default=1024, type=int,
                        metavar='N', help='size of a total batch size, for simulating bigger batches')
    parser.add_argument('--learning-rate', type=float, default=1.024,
                        help='learning rate for the optimizer')
    parser.add_argument('--momentum', default=0.875, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov momentum, default: false)')
    parser.add_argument('--weight-decay', type=float, default=3.0517578125e-05,
                        help='weight decay factor for the optimizer')
    parser.add_argument('--bn-weight-decay', default=0.0, type=float,
                        help='weight decay on BN (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs to train')
    parser.add_argument('--train-log-freq', type=int, default=100,
                        help='The frequency (global step) to log the metrics during the training process')
    parser.add_argument('--ckpt-freq', type=int, default=50,
                        help='The frequency (epoch) to save the checkpoint')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='The output directory to save the checkpoint and training log.')
    parser.add_argument('--lr-schedule', default='linear', type=str, metavar='SCHEDULE',
                        choices=['step', 'linear', 'cosine', 'step_prune'])
    parser.add_argument('--warmup', default=16, type=int,
                        metavar='E', help='number of warmup epochs')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        metavar='S', help='label smoothing')
    parser.add_argument('--mixup', default=0.0, type=float,
                        metavar='ALPHA', help='mixup alpha')

    parser.add_argument('--no-prune', action='store_true', help='Perform pruning.')
    parser.add_argument('--prune-mode', type=str, default='exp2',
                        help='The mode for scheduling the neuron number to prune for each epoch.'
                             'either to be exponential ("exp") way or linear ("linear") way.')
    parser.add_argument('--reg-conf', type=str, default='configs/resnet50.json',
                        help='The json file defining the regularization configuration.')
    parser.add_argument('--prune-ratio', type=float, default=0.3,
                        help='the ratio of neurons going to be pruned.')
    parser.add_argument('--reg-start-epoch', type=int, default=0,
                        help='the start epoch of gs regularization')
    parser.add_argument('--reg-end-epoch', type=int, default=1,
                        help='the end epoch of gs regularization')
    parser.add_argument('--method', type=int, default=26,
                        help='the method to rank the neurons during the pruning')
    parser.add_argument('--baseline-file', type=str, default=None,
                        help='Txt file recoding the baseline accuracy.')
    parser.add_argument('--prune-per-epoch', action='store_true',
                        help='if true, perform pruning at the end of each epoch. '
                             'Otherwise, prune every 40 iterations and finishes in 30 steps for imagenet')
    parser.add_argument('--lut-file', default=None, type=str,
                        help='the path of the latency lookup table file.')
    parser.add_argument('--lut-bs', default=256, type=int,
                        help='the batch size used when creating the latency lookup table.')

    parser.add_argument('--pre-trained', type=str, default=None,
                        help='the path of a pre-trained model if want to start from some checkpoint')
    parser.add_argument('--mask', type=str, default=None,
                        help='the mask file')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='the start epoch of training (for restart)')

    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--static-loss-scale', type=float, default=128,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                             '--static-loss-scale.')

    parser.add_argument('--amp', action='store_true',
                        help='Run model AMP (automatic mixed precision) mode.')

    parser.add_argument("--local-rank", default=0, type=int)

    parser.add_argument('--seed', default=10, type=int,
                        help='random seed used for np and pytorch')

    parser.add_argument('--step-size', default=-1, type=int, help='step pruning for latency aware pruning')
    parser.add_argument('--mu', default=6e-4, type=float, help='the scalar for latency aware importance')

    parser.add_argument('--target-latency', default=None, type=float, help='the targeted latency')

    parser.add_argument('--clip', default=None, type=float)

    # parser.add_argument('--exp-name', type=str, default=None, help='name of the experiment')
    parser.add_argument('--pulp', action='store_true', help='Use pulp optimizer.')
    parser.add_argument('--pyomo', action='store_true', help='Use pyomo optimizer.')
    parser.add_argument('--no_blockprune', action='store_true', help='Enforce No block pruning. use --mgp and --mdp for layerwise representation.')
    parser.add_argument('--oneshot', action='store_true', help='Perform oneshot pruning.')
    parser.add_argument('--blocks_num', default=None, type=int, help='use a bounded number of blocks')
    
    parser.add_argument('--mgp', action='store_true', help='Use Multi-Granularity Pruning.')
    parser.add_argument('--mdlm', action='store_true', help='Use Multi-Dimensional Latency Modeling.')
    parser.add_argument('--mdp', action='store_true', help='run MDP (Multi-Dimensional Pruning).')
    return parser
