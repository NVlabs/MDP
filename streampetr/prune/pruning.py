import torch
import math
from collections import defaultdict
import pickle
import numpy as np
imp_prune_counter = 0
##############################################
# #################  method   ################
# -1: globally random pruning
# -2: layer-wise uniform pruning, the neurons in each layer are ranked by importance calculated as method 8
# -3: layer-wise uniform pruning, the neurons in each layer are ranked by l2 norm as method 0/1
# 0: magnitude-based method, rank by l2 norm and apply reg to the layers
# 1: magnitude-based method, rank by l2 norm and just mask pruned neurons, no weight update
# 2: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by sum(abs(w*w.grad)), if gate place, w is the gate parameter
# 3: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by abs(sum(w*w.grad)), if gate place, w is the gate parameter
# 4: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by sum((w*w.grad)^2), if gate place, w is the gate parameter
# 5: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by (sum(w*w.grad))^2, if gate place, w is the gate parameter
# 6: magnitude-based method, rank by norm*bn_gamma/bn_dev and apply reg to the layers. Not recommended!
# 7: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by sum(abs(w*w.grad)), w is the bn weights and bias, if gate place, w is the gate parameter
# 8: importance-based method, rank by importance and mask pruned neurons
#    importance calculated by abs(sum(w*w.grad)), w is the bn weights and bia, if gate place, w is the gate parameter
# 15: combine magnitude and gradient-importance together, basically is to use the importance as a scalar of
#     the regularization item, importance is divided by len(group)
# 16: combine magnitude and gradient-importance together, basically is to use the importance as a scalar of
#     the regularization item, importance is not divided by len(group)
# 9: latency-aware, combined with importance from method 8
#    more details: calculate a multiplier from latency change, scale the importance by the multiplier
# 10: latency-aware, combined with importance from method 1
#    more details: calculate a multiplier from latency change, scale the importance by the multiplier


def set_prune_target(total_conv_num, prune_ratio, gs_start, gs_end, mode='exp'):
    """
        schedule the target number of neurons to prune for each epoch
    Returns:
        to_prune: List with size of total epochs.
                  Each item is the number of neurons to prune at the corresponding epoch.
    """
    total_prune_num = int(total_conv_num * prune_ratio)
    if mode == 'exp':
        to_prune = [math.exp(x/20.0) for x in range(0, gs_end-gs_start)]
        scale = total_prune_num / sum(to_prune)
        to_prune = [int(x*scale) for x in to_prune][::-1]
    elif mode == 'exp2':  # exp schedule proposed in FORCE
        final_remain = total_conv_num - total_prune_num
        kt = [0 for _ in range(gs_end - gs_start + 1)]
        T = gs_end - gs_start
        for t in range(0, gs_end - gs_start + 1):
            alpha = t / T
            kt[t] = math.exp(alpha * math.log(final_remain) + (1 - alpha) * math.log(total_conv_num))
        to_prune = [int(kt[t] - kt[t + 1]) for t in range(0, gs_end - gs_start)]
    else:  # linear mode
        to_prune = [total_prune_num//(gs_end-gs_start) for _ in range(0, gs_end-gs_start)]
    remain = total_prune_num - sum(to_prune)
    for i in range(remain):
        to_prune[i] += remain//abs(remain)
    assert sum(to_prune) == total_prune_num
    print(to_prune)
    return to_prune


def hybrid_norm_importance(layers, groups, neuron_metric, method):
    importance_sum = 0
    for group in groups:
        importance_sum += torch.sum(neuron_metric[group]).item()
    for group in groups:
        l2_norm = calc_l2_norm(layers, group)
        neuron_metric[group] = torch.mul(l2_norm, neuron_metric[group]/importance_sum)
        if method == 15:
            neuron_metric[group] /= len(group)


def update_neuron_metric_hahp_fusedconvkbn(layers, groups, layer_bn=None, layer_gate=None, method=26, neuron_metric={}):
    for group in groups:
        # We only have one layer in the group
        # Do nomarl storing and metric update
        if len(group) == 1:
            importance_score = calc_importance_fused_convkbn(layers, group, layer_bn, layer_gate, method)
            layer_name = group[0]
            if layer_name not in neuron_metric:
                neuron_metric[layer_name] = importance_score
            else:
                neuron_metric[layer_name] += importance_score
                # self._neuron_metric_update_counter[layer_name] += 1
        # For many layers in the group
        # Rescale the absolute values to split into different layers
        else:
            # Our target is to meet the following total combined importance value
            correct_importance_score = calc_importance_fused_convkbn(layers, group, layer_bn, layer_gate, method)
            # Get the total absolute importance, as denominator of our ratio
            abs_importance = 0
            for layer_name in group:
                layer = layers[layer_bn[layer_name]]
                abs_importance_score = abs(layer.weight.data)
                abs_importance += abs_importance_score
            abs_importance *= 10000
            for layer_name in group:
                layer = layers[layer_bn[layer_name]]
                # Compute absolute importance for the current layer, as numerator of our ratio
                abs_importance_score = abs(layer.weight.data)
                abs_importance_score *= 10000
                # Scale
                correct_importance_score_split = correct_importance_score * (abs_importance_score / abs_importance)
                if layer_name not in neuron_metric:
                    neuron_metric[layer_name] = correct_importance_score_split
                else:
                    neuron_metric[layer_name] += correct_importance_score_split


def update_neuron_metric_hahp(layers, groups, layer_bn=None, layer_gate=None, method=26, neuron_metric={}):
    for group in groups:
        # We only have one layer in the group
        # Do nomarl storing and metric update
        if len(group) == 1:
            for layer_name in group:
                tmp_importance = 0
                layer = layers[layer_bn[layer_name]]
                tmp_importance += (layer.weight.data * layer.weight.grad.data +
                                layer.bias.data * layer.bias.grad.data)
                importance_score = tmp_importance.abs().detach()

                if layer_name not in neuron_metric:
                    neuron_metric[layer_name] = importance_score
                else:
                    neuron_metric[layer_name] += importance_score
                # self._neuron_metric_update_counter[layer_name] += 1
        # For many layers in the group
        # Rescale the absolute values to split into different layers
        else:
            # Our target is to meet the following total combined importance value
            correct_importance_score = calc_importance(layers, group, layer_bn, layer_gate, method)
            # if 'module.layer1.0.downsample.0' in group:
            #     # print(correct_importance_score)
            #     global imp_prune_counter
            #     with open(f"per_step_importance_score/latv2_hahp_{imp_prune_counter}_block1.pkl", 'wb') as f:
            #         pickle.dump(correct_importance_score, f)
            #     imp_prune_counter += 1
            # Simple testing code to debug
            # for layer_name in group:
            #     layer = layers[layer_bn[layer_name]] 
            #     if layer_name not in neuron_metric:
            #         neuron_metric[layer_name] = correct_importance_score / len(group)
            #     else:
            #         neuron_metric[layer_name] += correct_importance_score / len(group)
            # Get the total absolute importance, as denominator of our ratio
            abs_importance = 0
            for layer_name in group:
                layer = layers[layer_bn[layer_name]]
                abs_importance_score = abs(layer.weight.data * layer.weight.grad.data + layer.bias.data * layer.bias.grad.data)
                abs_importance += abs_importance_score
            abs_importance *= 10000
            if 0 in abs_importance:
                eps = (abs_importance == 0).float()
                abs_importance += eps
            for layer_name in group:
                layer = layers[layer_bn[layer_name]]
                # Compute absolute importance for the current layer, as numerator of our ratio
                abs_importance_score = abs(layer.weight.data * layer.weight.grad.data + layer.bias.data * layer.bias.grad.data)
                abs_importance_score *= 10000
                # Scale
                correct_importance_score_split = correct_importance_score * (abs_importance_score / abs_importance)
                if layer_name not in neuron_metric:
                    neuron_metric[layer_name] = correct_importance_score_split
                else:
                    neuron_metric[layer_name] += correct_importance_score_split


def update_neuron_metric(layers, groups, layer_bn=None, layer_gate=None, method=0, neuron_metric={},
                         pre_group=None, aft_group_list=None, step_size=4, mu=0.5, group_mask=None,
                         fmap_table=None, lookup_table=None, last_iter=False, imp_convk=False, imp_fused_convk=False):
    for group in groups:
        if method == 0 or method == 1 or method == -1 or method == -3 or method == 10 or method == 24:
            neuron_metric[group] = calc_l2_norm(layers, group)
        elif method == 6:
            neuron_metric[group] = calc_bn_scaled_l2_norm(layers, group, layer_bn)
        elif method == 11:
            importance_score = calc_hessian_importance(layers, group, layer_bn)
            if group not in neuron_metric:
                neuron_metric[group] = importance_score
            else:
                neuron_metric[group] += importance_score
        else:
            if imp_convk:
                importance_score = calc_importance_convk(layers, group, layer_bn, layer_gate, method)
            elif imp_fused_convk:
                importance_score = calc_importance_fused_convkbn(layers, group, layer_bn, layer_gate, method, fmap_table)
            else:
                importance_score = calc_importance(layers, group, layer_bn, layer_gate, method)
                # if 'module.layer1.0.downsample.0' in group:
                #     # print(correct_importance_score)
                #     global imp_prune_counter
                #     with open(f"per_step_importance_score/latv2_{imp_prune_counter}_block1.pkl", 'wb') as f:
                #         pickle.dump(importance_score, f)
                #     imp_prune_counter += 1

            if group not in neuron_metric:
                neuron_metric[group] = importance_score
            else:
                neuron_metric[group] += importance_score
        if last_iter:
            if method == 9 or method == 10:
                group_latency_change, max_group_latency_change = get_group_latency_change(layers, groups, pre_group, aft_group_list, step_size, group_mask, fmap_table, lookup_table)
                for group, importance_score in neuron_metric.items():
                    latency_change_multiplier = mu * (group_latency_change[tuple(group)] / max_group_latency_change)
                    latency_change_multiplier = 1. - min(max(latency_change_multiplier, 0.0), 1.0)
                    neuron_metric[group] = importance_score * latency_change_multiplier


def calc_l2_norm(layers, group):
    layer = layers[group[0]]
    group_weight = layer.weight.data.view(layer.weight.size(0), -1)
    for layer_name in group[1:]:
        layer = layers[layer_name]
        group_weight = torch.cat((group_weight,
                                 layer.weight.data.view(layer.weight.size(0), -1)),
                                 dim=1)
    group_norm = group_weight.norm(dim=1).detach() / math.sqrt(group_weight.size(1))
    return group_norm


def calc_bn_scaled_l2_norm(layers, group, layer_bn):
    eps = 1e-05
    layer = layers[group[0]]
    bn_layer = layers[layer_bn[group[0]]]
    group_weight = (layer.weight.data.view(layer.weight.size(0), -1) *
                    (bn_layer.weight.data / torch.sqrt(bn_layer.running_var.data + eps)).view(bn_layer.weight.size(0), -1))
    for layer_name in group[1:]:
        layer = layers[layer_name]
        bn_layer = layers[layer_bn[layer_name]]
        group_weight = torch.cat((group_weight,
                                 (layer.weight.data.view(layer.weight.size(0), -1)) *
                                  (bn_layer.weight.data / torch.sqrt(bn_layer.running_var.data + eps)).view(bn_layer.weight.size(0), -1)),
                                 dim=1)
    group_norm = group_weight.norm(dim=1).detach() / math.sqrt(group_weight.size(1))
    return group_norm


def calc_importance_convk(layers, group, layer_bn=None, layer_gate=None, method=2):
    if layer_gate is not None:
        gate_layer = layers[layer_gate[group[0]]]
        weights = gate_layer.weight.data.view(-1, 1)
        grads = gate_layer.weight.grad.data.view(-1, 1)
    else:
        layer = layers[group[0]]
        weights = layer.weight.data.view(layer.weight.size(0), -1)
        # print(layer.weight.grad)
        grads = layer.weight.grad.data.view(layer.weight.size(0), -1)
        for layer_name in group[1:]:
            layer = layers[layer_name]
            weights = torch.cat((weights,
                                    layer.weight.data.view(layer.weight.size(0), -1)), dim=1)
            grads = torch.cat((grads,
                                layer.weight.grad.data.view(layer.weight.size(0), -1)), dim=1)
    # print("Imp Convk")
    importance_score = (weights * grads).detach().sum(dim=1).abs()
    return importance_score

def reweight_basedon_reorg(w_imp, fm_size):
    # return w_imp
    # print(torch.sum(w_imp))
    mask = torch.sum(w_imp.abs(), dim = [1,2,3]) > 0
    # print(torch.sum(mask))
    # print(fm_size)
    largest = fm_size ** 2
    weights = []
    cur = largest
    k_size = 3
    for i in range(k_size):
        weights.append(float(cur))
        cur -= (fm_size - i)
    # print(weights)
    scaling = [[weights[2], weights[1], weights[2]], [weights[1], weights[0], weights[1]], [weights[2], weights[1], weights[2]]]
    scaling = torch.tensor(scaling)
    scaling = scaling.to(w_imp.device)
    # Normalize
    scaling = scaling / sum(weights) * 3.0
    # print(w_imp.shape, scaling.shape)
    # print(scaling)
    # print(scaling)
    # print(torch.sum(1 - mask.to(torch.float)))
    w_imp[mask] *= scaling
    # print(torch.sum(w_imp))
    return w_imp

def calc_importance_fused_convkbn(layers, group, layer_bn=None, layer_gate=None, method=2):
    
    bn_layer = layers[layer_bn[group[0]]]
    layer = layers[group[0]]
    w_bn = torch.diag(bn_layer.weight.data.div(torch.sqrt(bn_layer.eps+bn_layer.running_var))).cuda()
    w_conv = layer.weight.data.clone()
    w_conv = w_conv * layer.weight.grad.data
    w_conv = w_conv.view(w_conv.size(0), -1).cuda()
    w_imp = torch.mm(w_bn, w_conv)
    # w_imp = w_imp.sum(dim=1) * bn_layer.weight.grad.data
    w_imp = w_imp.sum(dim=1)
    # b_imp = bn_layer.bias.data * bn_layer.bias.grad.data
    # tmp_importance = (w_imp + b_imp)
    # b_imp = bn_layer.bias.data * bn_layer.bias.grad.data
    tmp_importance = w_imp

    for layer_name in group[1:]:
        bn_layer = layers[layer_bn[layer_name]]
        layer = layers[layer_name]
        w_bn = torch.diag(bn_layer.weight.data.div(torch.sqrt(bn_layer.eps+bn_layer.running_var))).cuda()
        w_conv = layer.weight.data.clone()
        w_conv = w_conv * layer.weight.grad.data
        w_conv = w_conv.view(w_conv.size(0), -1).cuda()
        w_imp = torch.mm(w_bn, w_conv)
        w_imp = w_imp.sum(dim=1)
        tmp_importance += w_imp

    importance_score = tmp_importance.abs().detach()
    # print(group)
    # print(torch.sum(importance_score))
    if torch.isnan(torch.sum(importance_score)):
        print(group)
        print("Importance is NaN.")
        importance_score = torch.zeros_like(importance_score)
    # if np.nan in importance_score:
    #     print(group, importance_score)
    
    return importance_score

def calc_importance(layers, group, layer_bn=None, layer_gate=None, method=2):
    if method in [2, 3, 4, 5] or layer_bn is None:
        if layer_gate is not None:
            gate_layer = layers[layer_gate[group[0]]]
            weights = gate_layer.weight.data.view(-1, 1)
            grads = gate_layer.weight.grad.data.view(-1, 1)
        else:
            layer = layers[group[0]]
            weights = layer.weight.data.view(layer.weight.size(0), -1)
            grads = layer.weight.grad.data.view(layer.weight.size(0), -1)
            for layer_name in group[1:]:
                layer = layers[layer_name]
                weights = torch.cat((weights,
                                     layer.weight.data.view(layer.weight.size(0), -1)), dim=1)
                grads = torch.cat((grads,
                                   layer.weight.grad.data.view(layer.weight.size(0), -1)), dim=1)
        if method == 2:
            importance_score = (weights * grads).detach().abs().sum(dim=1)
        elif method == 3 or layer_bn is None:
            importance_score = (weights * grads).detach().sum(dim=1).abs()
        elif method == 4:
            importance_score = (weights * grads).detach().pow(2).sum(dim=1)
        elif method == 5:
            importance_score = (weights * grads).detach().sum(dim=1).pow(2)
        else:
            raise NotImplementedError
    elif method == 7:
        layer = layers[layer_bn[group[0]]]
        tmp_importance = (layer.weight.data * layer.weight.grad.data).abs() + (layer.bias.data * layer.bias.grad.data).abs()
        for layer_name in group[1:]:
            layer = layers[layer_bn[layer_name]]
            tmp_importance += ((layer.weight.data * layer.weight.grad.data).abs() +
                               (layer.bias.data * layer.bias.grad.data).abs())
        importance_score = tmp_importance.detach()
    elif method in [8, -2, 15, 16, 9, 20, 21, 22, 23, 26]:
        layer = layers[layer_bn[group[0]]]
        tmp_importance = layer.weight.data * layer.weight.grad.data + layer.bias.data * layer.bias.grad.data
        for layer_name in group[1:]:
            layer = layers[layer_bn[layer_name]]
            tmp_importance += (layer.weight.data * layer.weight.grad.data +
                               layer.bias.data * layer.bias.grad.data)
        importance_score = tmp_importance.abs().detach()
    else:
        raise NotImplementedError
    return importance_score


def calc_hessian_importance(layers, group, layer_bn=None):
    layer = layers[layer_bn[group[0]]]
    importance_score = -(layer.weight.data * layer.weight.grad.data).detach()
    if torch.distributed.is_initialized():
        # handle different GPUs having different stats:
        torch.distributed.all_reduce(importance_score.data)
    return importance_score


def get_group_latency_change(layers, groups, pre_group, aft_group_list, step_size, group_mask=None, fmap_table=None, lookup_table=None):
    max_group_latency_change = 0.
    group_latency_change = {}
    for group in groups:
        if group_mask is not None and group_mask != {}:
            mask = group_mask[group]
            active_neuron_num = torch.sum(mask.data).cpu().item()
        else:
            active_neuron_num = layers[group[0]].weight.size(0)
        ori_active_neuron_num = active_neuron_num
        active_neuron_num = find_closest_step(active_neuron_num, step_size)
        latency_change = 0
        # latency change caused by the neuron num change in the pruned layers
        for layer_name in group:
            pre_group_name = pre_group[layer_name]
            if group_mask is not None and group_mask != {}:
                pre_active_neuron_num = torch.sum(
                    group_mask[pre_group_name].data).cpu().item() if pre_group_name is not None else 3
            else:
                pre_active_neuron_num = layers[pre_group_name[0]].weight.size(0) if pre_group_name is not None else 3
            if pre_group_name is not None:
                pre_active_neuron_num = find_closest_step(pre_active_neuron_num, step_size)
            layer = layers[layer_name]
            k = layer.kernel_size[0]
            fmap = fmap_table[layer_name]
            stride = layer.stride[0]
            latency = get_latency(lookup_table, pre_active_neuron_num, active_neuron_num, k, fmap, stride)
            reduced_latency = get_latency(lookup_table, pre_active_neuron_num, max(active_neuron_num-step_size, 0), k, fmap, stride)
            layer_latency_change = (latency - reduced_latency)/step_size
            layer_latency_change = max(layer_latency_change, 0) * 0.8 + (0.2 * latency / ori_active_neuron_num if ori_active_neuron_num > 0 else 0.)
            latency_change += layer_latency_change
        # latency change on the following layers (the input size of the following layers will change)
        aft_groups = aft_group_list[group]
        for aft_group in aft_groups:
            if group_mask is not None and group_mask != {}:
                aft_active_neuron_num = torch.sum(group_mask[aft_group].data).cpu().item()
            else:
                aft_active_neuron_num = layers[aft_group[0]].weight.size(0)
            aft_active_neuron_num = find_closest_step(aft_active_neuron_num, step_size)
            k = layers[aft_group[0]].kernel_size[0]
            fmap = fmap_table[aft_group[0]]
            stride = layers[aft_group[0]].stride[0]
            latency = get_latency(lookup_table, active_neuron_num, aft_active_neuron_num, k, fmap, stride)
            reduced_latency = get_latency(lookup_table, max(active_neuron_num - step_size, 0), aft_active_neuron_num, k, fmap, stride)
            layer_latency_change = (latency - reduced_latency) / step_size
            layer_latency_change = max(layer_latency_change, 0) * 0.8 + (0.2 * latency / ori_active_neuron_num if ori_active_neuron_num > 0 else 0.)
            latency_change += layer_latency_change
        group_latency_change[group] = latency_change
        max_group_latency_change = max(max_group_latency_change, latency_change)
    return group_latency_change, max_group_latency_change


def find_closest_step(num, step_size):
    steps = num // step_size
    low = steps * step_size
    high = (steps + 1) * step_size
    if num - low < high - num:
        return low
    else:
        return high


def get_latency(lookup_table, cin=16, cout=16, k=3, fmap=128, stride=2):
    batch_size = 256
    if cin <= 0 or cout <= 0:
        return 0
    key = '_'.join(
        [str(batch_size), str(int(cin)), str(int(cout)), str(fmap), str(k), str(stride)])
    lat_speed = lookup_table[key]
    return lat_speed


def find_layers_to_prune(metric, group_mask, target_num, pre_pruned_num, pre_target_num):
    """
        Rank all the neurons(groups) over the whole network.
        Get the number of neurons to prune for each layer(group) and set the corresponding penalty strength
    Args:
        metric: dict, stores the importance measurement of each group
                the key is the group name, the value is the corresponding importance measurement
        target_num: int, the scheduled target number of neurons to prune in this epoch
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
    group_prune_numbers = defaultdict(int)  # to count the number of neurons to prune for each group
    for group, metric_value in metric.items():
        if group in group_mask:
            mask = group_mask[group]
            value_to_compare = metric_value[mask == 1.]
        else:
            group_mask[group] = torch.ones_like(metric_value)
            value_to_compare = metric_value
        all_compare_values.append(value_to_compare)
        index2name.extend([group for _ in range(int(value_to_compare.size(0)))])
    
    all_compare_values = torch.cat(all_compare_values)
    # take the neurons(groups) with the least rank
    value, prune_index = torch.topk(all_compare_values,
                                    min(updated_target_num, all_compare_values.size(0)),
                                    largest=False)
    print(value, prune_index)
    prune_num = 0
    for v, index in zip(value, prune_index):
        group = index2name[index]
        group_thresh[group] = v
        group_prune_numbers[group] += 1
        prune_num += len(group)
        if prune_num >= updated_target_num:
            break

    print(group_prune_numbers)
    return group_thresh


def find_layers_to_prune_latency_step(metric, group_mask, target_num, pre_pruned_num, pre_target_num):
    """
        Rank all the neurons(groups) over the whole network.
        Get the number of neurons to prune for each layer(group) and set the corresponding penalty strength
    Args:
        metric: dict, stores the importance measurement of each group
                the key is the group name, the value is the corresponding importance measurement
        target_num: int, the scheduled target number of neurons to prune in this epoch
    Returns:
        layer_thresh: dict, stores the penalty strength of each group
                      the key is the group name, the value is the penalty strength
    """
    # update the target number if previously prunes is slightly different from scheduled
    updated_target_num = target_num - (pre_pruned_num - pre_target_num)
    if updated_target_num <= 0:
        pre_target_num += target_num
        return {}

    step_size = 8
    all_compare_values = []  # to store the value for compare in order to rank the groups
    index2name = []
    threshes = {}
    group_thresh = {}
    group_prune_numbers = defaultdict(int)  # to count the number of neurons to prune for each group
    for group, metric_value in metric.items():
        if group in group_mask:
            mask = group_mask[group]
            value_remained = metric_value[mask == 1.]
        else:
            group_mask[group] = torch.ones_like(metric_value)
            value_remained = metric_value
        if 'module.conv1' in group and int(value_remained.size(0)) == 8:
            continue
        sorted_value, indices = torch.sort(value_remained)
        sorted_value = sorted_value.view(-1, step_size)
        value_to_compare = sorted_value.sum(dim=1)
        threshes[group] = list(sorted_value[:, -1].cpu().numpy())
        all_compare_values.append(value_to_compare)
        index2name.extend([group for _ in range(int(value_to_compare.size(0)))])

    all_compare_values = torch.cat(all_compare_values)
    # take the neurons(groups) with the least rank
    _, prune_index = torch.topk(all_compare_values,
                                    min(updated_target_num//step_size+1, all_compare_values.size(0)),
                                    largest=False)

    pruned_neuron_index = defaultdict(list)  # {group_name: neuron_index}
    prune_num = 0
    first_channel_num = int(group_mask[tuple(['module.conv1'])].sum().item())
    first_downsample_num = int(group_mask[tuple(["module.layer1.0.downsample.0", "module.layer1.0.conv3", "module.layer1.1.conv3", "module.layer1.2.conv3"])].sum().item())
    for index in prune_index:
        group = index2name[index]
        if ('module.conv1' in group and first_channel_num == 8) or ('module.layer1.0.downsample.0' in group and first_downsample_num == 8):
            continue
        if 'module.conv1' in group:
            first_channel_num -= 8
        if 'module.layer1.0.downsample.0' in group:
            first_downsample_num -= 8
        group_thresh[group] = threshes[group].pop(0)
        group_prune_numbers[group] += step_size
        prune_num += (len(group)*step_size)
        all_values = metric[group][group_mask[group] == 1.]
        pruned_values = all_values[all_values <= group_thresh[group]]
        for v in pruned_values:
            actual_neuron_index = (metric[group] == v).nonzero()[0].item()
            pruned_neuron_index[group].append(actual_neuron_index)
        if prune_num >= updated_target_num:
            break

    print(group_prune_numbers)
    return group_thresh


def random_pruning_global(layers, metric, target_num, pre_pruned_num, pre_target_num, group_mask):
    updated_target_num = target_num - (pre_pruned_num - pre_target_num)
    if updated_target_num <= 0:
        pre_target_num += target_num
    import numpy as np

    remained = []
    total_num = 0
    index2group = []
    index2index = []
    for group in metric.keys():
        neuron_num = layers[group[0]].weight.size(0)
        if group not in group_mask:
            group_mask[group] = torch.ones_like(metric[group])
            remained.extend([total_num+i for i in range(neuron_num)])
        else:
            tmp = (group_mask[group] > 0).nonzero().squeeze().cpu().numpy()
            remained.extend(list(tmp + total_num))
        total_num += neuron_num
        index2group.extend([group for _ in range(neuron_num)])
        index2index.extend([i for i in range(neuron_num)])

    np.random.seed(None)
    selected = np.random.choice(np.array(remained), updated_target_num, replace=False)
    prune_num = 0
    actual_pruned = []
    new_group_mask = {}
    for idx in selected:
        group_name = index2group[idx]
        if group_name not in new_group_mask:
            new_group_mask[group_name] = torch.ones_like(metric[group_name])
        new_group_mask[group_name][index2index[idx]] = 0
        prune_num += len(group_name)
        actual_pruned.append(idx)
        if prune_num >= updated_target_num:
            break
    return prune_num, new_group_mask


def pruning_uniform(metric, group_mask, target_num, pre_pruned_num, pre_target_num):
    updated_target_num = target_num - (pre_pruned_num - pre_target_num)
    if updated_target_num <= 0:
        pre_target_num += target_num
        return {}
    all_compare_values = []  # to store the value for compare in order to rank the groups
    index2name = []
    group_thresh = {}
    group_prune_numbers = defaultdict(int)  # to count the number of neurons to prune for each group
    group_remain = {}
    for group, metric_value in metric.items():
        if group in group_mask:
            mask = group_mask[group]
            value_to_compare = metric_value[mask == 1.]
        else:
            group_mask[group] = torch.ones_like(metric_value)
            value_to_compare = metric_value
        group_remain[group] = value_to_compare.size(0)
        if value_to_compare.size(0) <= metric_value.size(0)*0.5:
            continue
        all_compare_values.append(value_to_compare)
        index2name.extend([group for _ in range(int(value_to_compare.size(0)))])

    all_compare_values = torch.cat(all_compare_values)
    # take the neurons(groups) with the least rank
    value, prune_index = torch.topk(all_compare_values,
                                    all_compare_values.size(0),
                                    largest=False)

    prune_num = 0
    for v, index in zip(value, prune_index):
        group = index2name[index]
        if group_remain[group] <= metric[group].size(0)*0.5:
            continue
        group_remain[group] -= 1
        group_thresh[group] = v
        group_prune_numbers[group] += 1
        prune_num += len(group)
        if prune_num >= updated_target_num:
            break

    print(group_prune_numbers)
    return group_thresh


def mask_neurons2(layers, group_mask, layer_bn=None, layer_gate=None):
    for group, mask in group_mask.items():
        gate_name = None if layer_gate is None else layer_gate[group[0]]
        if gate_name is not None:
            gate_layer = layers[gate_name]
            gate_layer.weight.data.mul_(mask)
        for layer_name in group:
            layer = layers[layer_name]
            layer.weight.data.mul_(mask.view(-1, 1, 1, 1))
            if layer.bias is not None:
                layer.bias.data.mul_(mask)
            # mask corresponding bn
            if layer_bn is not None:
                bn_layer = layers[layer_bn[layer_name]]
                bn_layer.weight.data.mul_(mask)
                bn_layer.bias.data.mul_(mask)
                bn_layer.running_mean.data.mul_(mask)
                bn_layer.running_var.data.mul_(mask)


def apply_pruning(layers, layer_bn, layer_gate, metric, target_num, pre_pruned_num, pre_target_num, method, group_mask):
    """
        Apply the GS/CL regularization to 'prune' the network.
    Args:
        metric: dict, stores the importance measurement of each group
                the key is the group name, the value is the corresponding importance measurement
        target_num: int, the scheduled target number of neurons to prune in this epoch
    Returns:
        applied: True/False, whether applied regularization at this epoch
        pruned: True/False, if there are neurons pruned at this epoch or not
        reg_strength: dict, the key is the group name, the value is the applied regularization strength
    """
    if method == -1:
        total_pruned_num, new_group_mask = random_pruning_global(layers, metric, target_num, pre_pruned_num, pre_target_num, group_mask)
        for group in new_group_mask.keys():
            group_mask[group] = group_mask[group] * new_group_mask[group]
        mask_neurons2(layers, group_mask, layer_bn, layer_gate)
        return total_pruned_num
    if method == -2 or method == -3:
        group_thresh = pruning_uniform(metric, group_mask, target_num, pre_target_num, pre_target_num)
    elif method == 9 or method == 10:
        group_thresh = find_layers_to_prune_latency_step(metric, group_mask, target_num, pre_pruned_num,
                                                         pre_target_num)
    else:
        # get the penalty thresh for each group
        group_thresh = find_layers_to_prune(metric, group_mask, target_num, pre_pruned_num, pre_target_num)
    # apply the regularization and 'prune' the neurons
    total_pruned_num = 0
    for group, thresh in group_thresh.items():
        ori_channel_num = int(torch.sum(group_mask[group]).item())
        if method not in [0, 6, 15, 16]:
            mask = mask_neurons(layers, group, metric[group], thresh,
                                layer_bn=layer_bn,
                                gate_name=None if layer_gate is None else layer_gate[group[0]])
        else:
            mask = GS_reg(layers, group, metric[group], thresh,
                          layer_bn=layer_bn,
                          gate_name=None if layer_gate is None else layer_gate[group[0]])
        # update mask
        group_mask[group] = group_mask[group] * mask
        if method == 9 or method == 10:
            if int(torch.sum(group_mask[group]).item()) % 8 != 0:
                total_num = int(group_mask[group].size(0) - torch.sum(group_mask[group]).item())
                _, indexes = torch.topk(metric[group], total_num, largest=False)
                fix_count = 0
                for i in indexes.cpu().numpy()[-1::-1]:
                    if group_mask[group][i] == 0.:
                        group_mask[group][i] = 1.
                        fix_count += 1
                    if int(torch.sum(group_mask[group]).item()) % 8 == 0:
                        break

        cur_channel_num = int(torch.sum(group_mask[group]).item())
        pruned_channel = ori_channel_num - cur_channel_num
        total_pruned_num += pruned_channel*len(group)
        print('*** Group {}: {} channels / {} neurons are pruned at current step. '
              '{} channels / {} neurons left. ***'.format(group,
                                                          pruned_channel,
                                                          pruned_channel*len(group),
                                                          cur_channel_num,
                                                          cur_channel_num*len(group)))

    return total_pruned_num


def mask_neurons(layers, group, metric_value, thresh, layer_bn=None, gate_name=None):
    mask = (metric_value > thresh).type(metric_value.type())
    if gate_name is not None:
        gate_layer = layers[gate_name]
        gate_layer.weight.data.mul_(mask)
    for layer_name in group:
        layer = layers[layer_name]
        layer.weight.data.mul_(mask.view(-1, 1, 1, 1))
        if layer.bias is not None:
            layer.bias.data.mul_(mask)
        # mask corresponding bn
        if layer_bn is not None:
            bn_layer = layers[layer_bn[layer_name]]
            bn_layer.weight.data.mul_(mask)
            bn_layer.bias.data.mul_(mask)
            bn_layer.running_mean.data.mul_(mask)
            bn_layer.running_var.data.mul_(mask)
    return mask


def GS_reg(layers, group, metric_value, thresh, layer_bn=None, gate_name=None):
    mask = (metric_value > thresh).type(metric_value.type())
    for layer_name in group:
        layer_weight = layers[layer_name].weight
        layer_bias = layers[layer_name].bias
        W_update, layer_mask = soft_threshold(layer_weight,
                                              metric_value,
                                              thresh)
        # update conv weights
        layer_weight.data.copy_(W_update)
        if layer_bias is not None:
            layer_bias.data.mul_(layer_mask)
        # mask corresponding bn
        if layer_bn is not None:
            bn_layer = layers[layer_bn[layer_name]]
            bn_layer.weight.data.mul_(layer_mask)
            bn_layer.bias.data.mul_(layer_mask)
            bn_layer.running_mean.data.mul_(layer_mask)
            bn_layer.running_var.data.mul_(layer_mask)

        assert layer_mask[mask==0.].sum().item() == 0.
    if gate_name is not None:
        gate_data = layers[gate_name].weight.data.clone().detach()
        gate_data.mul_(mask)
        layers[gate_name].weight.data.copy_(gate_data.data)
    return mask


def soft_threshold(weight, group_norm, thresh):
    """
        Apply the soft-threshold operator to update the weights
    Args:
        weight: conv weight tensor
        strength: the penalty strength of the regularization
        layer_names: a list of conv names
        metric: dict, stores the importance measurement of each group
    Returns:
        W_update: the updated conv weight tensor
        layer_mask: the resulted mask tensor after pruning
    """
    eps = 1e-15
    eps_zero = 1e-10

    W = weight.view(weight.size(0), -1)

    # calculate the updated weights
    norm_modif = torch.div(torch.max(group_norm - thresh, torch.zeros_like(group_norm)), group_norm + eps)
    W_update = torch.mul(W, norm_modif.view(-1, 1))
    # get the mask
    WkNorm = W_update.norm(dim=1)
    layer_mask = (WkNorm > eps_zero).detach().type(weight.data.type())

    W_update = torch.mul(W_update, layer_mask.view(-1, 1))
    W_update = torch.reshape(W_update, weight.size())

    return W_update, layer_mask
