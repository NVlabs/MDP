import torch
import math
from collections import defaultdict
from utils.utils import count_non_zero_neurons

##############################################
# #################  method   ################
# 1: magnitude-based method, rank by l2 norm and just mask pruned neurons, no weight update
# 2: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by abs(w*w.grad), if gate place, w is the gate parameter


def update_weight_metric(layers, groups, layer_bn=None, layer_gate=None, method=1, weight_metric={}):
    for group in groups:
        if method == 1:
            weight_metric[group] = calc_l2_magnitude(layers, group)
        else:
            importance_score = calc_importance(layers, group, layer_bn, layer_gate, method)
            if group not in weight_metric:
                weight_metric[group] = importance_score
            else:
                weight_metric[group] += importance_score


def calc_l2_magnitude(layers, group):
    layer = layers[group]
    group_weight = layer.weight.data.view(-1)
    group_abs = group_weight.abs().detach().view_as(layer.weight)
    return group_abs


def calc_importance(layers, group, layer_bn=None, layer_gate=None, method=2):
    if layer_gate is not None:
        gate_layer = layers[layer_gate[group]]
        weights = gate_layer.weight.data.view(-1)
        grads = gate_layer.weight.grad.data.view(-1)
        importance_score = (weights * grads).detach().abs().view_as(gate_layer.weight)
    else:
        layer = layers[group]
        weights = layer.weight.data.view(-1)
        grads = layer.weight.grad.data.view(-1)
        importance_score = (weights * grads).detach().abs().view_as(layer.weight)
    return importance_score


def find_layers_to_prune(metric, group_mask, target_num, pre_pruned_num, pre_target_num, is_main=False):
    """
        Rank all the weights(groups) over the whole network.
        Get the number of weights to prune for each layer(group) and set the corresponding penalty strength
    Args:
        metric: dict, stores the importance measurement of each group
                the key is the group name, the value is the corresponding importance measurement
        group_mask: the mask for each group.
        target_num: int, the scheduled target number of neurons to prune in this epoch
        pre_pruned_num: int, the actual number of neurons to be pruned before this epoch
        pre_target_num: int, the scheduled number of neurons to be pruned before this epoch
        is_main: indicator if the method is currently part of the main thread.
    Returns:
        layer_thresh: dict, stores the penalty strength of each group
                      the key is the group name, the value is the penalty strength
    """
    # update the target number if previously prunes is slightly different from scheduled
    updated_target_num = target_num - (pre_pruned_num - pre_target_num)
    if updated_target_num <= 0:
        pre_target_num += target_num
        return {}
    all_compare_values = []  # to store the value for compare in order to rank the groups
    index2name = []
    group_thresh = {}
    group_prune_numbers = defaultdict(int)  # to count the number of weights to prune for each group
    for group, metric_value in metric.items():
        if group in group_mask:
            mask = group_mask[group]
            value_to_compare = metric_value[mask == 1.]
        else:
            group_mask[group] = torch.ones_like(metric_value)
            value_to_compare = metric_value
        value_to_compare = value_to_compare.view(-1)
        all_compare_values.append(value_to_compare)
        index2name.extend([group for _ in range(torch.numel(value_to_compare))])

    all_compare_values = torch.cat(all_compare_values)
    # take the weights(groups) with the least rank
    value, prune_index = torch.topk(all_compare_values,
                                    min(updated_target_num, torch.numel(all_compare_values)),
                                    largest=False)

    prune_num = 0
    for v, index in zip(value, prune_index):
        group = index2name[index]
        group_thresh[group] = v
        group_prune_numbers[group] += 1
        prune_num += 1
        if prune_num >= updated_target_num:
            break
    if is_main:
        print(group_prune_numbers)
    return group_thresh


def apply_pruning(layers, metric, target_num, pre_pruned_num, pre_target_num, group_mask, is_main=False):
    """
        Apply the regularization to 'prune' the network.
    Args:
        layers: the layers of the network
        metric: dict, stores the importance measurement of each group
                the key is the group name, the value is the corresponding importance measurement
        target_num: int, the scheduled target number of neurons to prune in this epoch
        pre_pruned_num: int, the actual number of neurons to be pruned before this epoch
        pre_target_num: int, the scheduled number of neurons to be pruned before this epoch
        group_mask: the mask for each group.
        is_main: indicator if the method is currently part of the main thread.
    Returns:
        total_pruned_num: Number of weights pruned.
    """
    # get the penalty thresh for each group
    group_thresh = find_layers_to_prune(metric, group_mask, target_num, pre_pruned_num, pre_target_num, is_main=is_main)

    # apply the regularization and 'prune' the neurons
    total_pruned_num = 0
    for group, thresh in group_thresh.items():
        ori_weight_num = int(torch.sum(group_mask[group]).item())
        mask = mask_neurons(layers, group, metric[group], thresh)
        # update mask
        group_mask[group] = group_mask[group] * mask

        cur_weight_num = int(torch.sum(group_mask[group]).item())
        pruned_weight = ori_weight_num - cur_weight_num
        total_pruned_num += pruned_weight
        if is_main:
            print('*** Group {}: {} weights are pruned at current step. '
                  '{} weights left. ***'.format(group,
                                                pruned_weight,
                                                cur_weight_num))

    return total_pruned_num


def mask_neurons(layers, group, metric_value, thresh):
    mask = (metric_value > thresh).type(metric_value.type())
    layer = layers[group]
    layer.weight.data.mul_(mask.view_as(layer.weight))
    return mask


def apply_two_to_four_pruning(layers, metric, group_mask, is_main=False):
    total_pruned_num = 0
    for group, metric_value in metric.items():
        group_mask[group] = torch.ones_like(metric_value)
        ori_weight_num = int(torch.sum(group_mask[group]).item())

        layer = layers[group]
        weight = layer.weight.detach()
        out_channels, in_channels, k_w, k_h = weight.size()
        weight = weight.permute(0, 2, 3, 1).reshape(-1, 4)  # (out_channels * k_w * k_h * in_channels // 4, 4)
        topk, indices = torch.topk(weight, 2)
        weight = torch.zeros_like(weight).scatter_(1, indices, topk)  # (out_channels * k_w * k_h * in_channels // 4, 4)
        weight = weight.view(out_channels, k_w, k_h, in_channels).permute(0, 3, 1, 2)  # (out_channels, in_channels, k_w, k_h)

        mask = (weight != 0.)  # (out_channels, in_channels, k_w, k_h)
        layer.weight.data.mul_(mask)
        group_mask[group] = group_mask[group] * mask

        cur_weight_num = int(torch.sum(group_mask[group]).item())
        pruned_weight = ori_weight_num - cur_weight_num
        total_pruned_num += pruned_weight

        if is_main:
            print('*** Group {}: {} weights are pruned at current step. '
                  '{} weights left. ***'.format(group,
                                                pruned_weight,
                                                cur_weight_num))

    return total_pruned_num

def apply_two_to_four_pruning_one_by_one(layers, group, metric, group_mask, is_main=False):
    total_pruned_num = 0
    metric_value = metric[group]
    group_mask[group] = torch.ones_like(metric_value)
    ori_weight_num = int(torch.sum(group_mask[group]).item())

    layer = layers[group]
    weight = layer.weight.detach()
    out_channels, in_channels, k_w, k_h = weight.size()
    weight = weight.permute(0, 2, 3, 1).reshape(-1, 4)  # (out_channels * k_w * k_h * in_channels // 4, 4)
    topk, indices = torch.topk(weight, 2)
    weight = torch.zeros_like(weight).scatter_(1, indices, topk)  # (out_channels * k_w * k_h * in_channels // 4, 4)
    weight = weight.view(out_channels, k_w, k_h, in_channels).permute(0, 3, 1, 2)  # (out_channels, in_channels, k_w, k_h)

    mask = (weight != 0.)  # (out_channels, in_channels, k_w, k_h)
    layer.weight.data.mul_(mask)
    group_mask[group] = group_mask[group] * mask

    cur_weight_num = int(torch.sum(group_mask[group]).item())
    pruned_weight = ori_weight_num - cur_weight_num
    total_pruned_num += pruned_weight

    if is_main:
        print('*** Group {}: {} weights are pruned at current step. '
              '{} weights left. ***'.format(group,
                                            pruned_weight,
                                            cur_weight_num))

    return total_pruned_num