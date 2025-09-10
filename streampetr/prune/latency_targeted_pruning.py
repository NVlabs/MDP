import torch
import math
import itertools
import numpy as np
from collections import defaultdict
# from utils.utils import count_non_zero_neurons
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
)
from pyomo.environ import *
import pickle
prune_counter = 0
##############################################
# #################  method   ################
# 20: latency_target_pruning, in each layer (group), certain number of channels are groupped together, 
#     solve knapsack problem to select neurons.
# 21: to compare with latency target pruning. In each layer (group), certain number of channels are
#     groupped together, prune neurons according to the ranking of the total neuron importance of group,
#     until the targeted latency is matched.
# 22: to compare with latency target pruning. In each layer (group), certain number of channels are 
#     groupped together, prune neurons according to the ranking of the average neuron importance of group,
#     unitl the targeted latency is matched.
# 23: latency-aware pruning, same as method 9, but the group size is different
# 26: latency_target_pruning, in each layer (group), certain number of channels are groupped together,
#     solve knapsack problem to select neurons. The latency contribution of each group is calculated
#     adaptively.


# CHANNEL_GROUP_SIZE = 32
import json

EPS = 1e-10
channel_group_size_dict = {}

# def load_group_size(arch):
#     global channel_group_size_dict
#     with open('./prune/group_size/{}_group_size.json'.format(arch), 'r') as f:
#         channel_group_size_dict = json.load(f)
def load_group_size(arch):
    global channel_group_size_dict
    with open('./prune/group_size/{}_group_size_32.json'.format(arch), 'r') as f:
        channel_group_size_dict = json.load(f)

def set_latency_prune_target(initial_latency, prune_start, prune_end, target_latency=None, latency_reduction_ratio=None, mode='exp'):
    """
        Schedule the pruning to achieve the target latency iteratively. 
    Args:
        initial_latency (float): the latency of the full dense model.
        prune_start (int): the step to start pruning.
        prune_end (int): the step to stop pruning.
        target_latency (float): the final latency after pruning that we are targeting for. 
            If not provided, it will be set according to latency reduction ratio.
        latency_reduction (float): the latency reduction ratio compared to the initial one.
            If target latency is provided, the reduction ratio will
            be ignored.
        mode (str): the mode for schedule. linear, exp, exp2 are supported.
    Returns:
        latency_limits (list): List with size of total (pruning) steps.
            Each item is the latency limit at the corresponding step.
    """
    # need to provide either target_latency or latency_reduction_ratio
    assert target_latency is not None or latency_reduction_ratio is not None
    # if target latency is nor provided, set it from reduction ratio
    if target_latency is None:
        target_latency = initial_latency * (1 - latency_reduction_ratio)
    latency_to_reduce = initial_latency - target_latency
    
    if mode == 'exp':
        to_prune = [math.exp(x/20.0) for x in range(0, prune_end-prune_start)]
        scale = latency_to_reduce / sum(to_prune)
        to_prune = [x*scale for x in to_prune[::-1]]
    elif mode == 'exp2':  # exp schedule proposed in FORCE
        kt = [0 for _ in range(prune_end-prune_start+1)]
        T = prune_end - prune_start
        for t in range(0, prune_end - prune_start + 1):
            alpha = t / T
            kt[t] = math.exp(alpha * math.log(target_latency) + (1 - alpha) * math.log(initial_latency))
        to_prune = [kt[t] - kt[t + 1] for t in range(0, prune_end - prune_start)]
    else:  # linear mode
        to_prune = [latency_to_reduce/(prune_end-prune_start) for _ in range(0, prune_end-prune_start)]
    
    latency_limits = []
    for item in to_prune:
        latency_limits.append((latency_limits[-1] if len(latency_limits) > 0 else initial_latency) - item)

    return latency_limits


def group_neurons_by_rank(metric, group_mask):
    """
        Group the neurons/channels with 8 neurons/channels per group to make the pruned structure GPU friendly.
    Args:
        metric (dict): the metric to measure the importance of the neurons
        group_mask (dict): the mask for each layer group indicating which channels have been zeroed.
    Returns:
        grouped_neruons (dict): indicates the layer group name, the channel index and the total importance score 
            of each neuron group.
    """
    grouped_neurons = {}  # {grouped_neuron_idx: {"layer_group_name": name, "channel_indices": [idx], "combined_importance": value}}

    grouped_neuron_idx = 0
    for group, metric_value in metric.items():
        if group in group_mask:
            mask = group_mask[group]
            value_remained = metric_value[mask == 1.]
            channels_remained = np.arange(mask.size(0))[mask.cpu().numpy() == 1.]
        else:
            group_mask[group] = torch.ones_like(metric_value)
            value_remained = metric_value
            channels_remained = np.arange(group_mask[group].size(0))
        # the count of remaining neurons should be a multiple of 8
        assert 'module.conv1' in group or int(value_remained.size(0)) % max([channel_group_size_dict[ln] for ln in group]) == 0

        sorted_values, sorted_indices = torch.sort(value_remained)
        # sorted_values = sorted_values.view(-1, CHANNEL_GROUP_SIZE)
        # sorted_indices = sorted_indices.view(-1, CHANNEL_GROUP_SIZE)
        sorted_values = sorted_values.view(-1, max([channel_group_size_dict[ln] for ln in group]))
        sorted_indices = sorted_indices.view(-1, max([channel_group_size_dict[ln] for ln in group]))
        combined_importance = sorted_values.sum(dim=1)
        if 'module.conv1' in group or ('module.features.conv_bn.conv' in group and int(value_remained.size(0)) <= 16):
#         if 'module.conv1' in group:
            combined_importance[:] = 10000  # set to large values to avoid pruning
        for i in range(sorted_indices.size(0)):
            grouped_neurons[grouped_neuron_idx] = {
                "layer_group_name": group,
                "channel_indices": [channels_remained[idx] for idx in sorted_indices[i]],
                "combined_importance": combined_importance[i].item()
            }
            grouped_neuron_idx += 1

        # importance scores are sorted in ascending order,
        # thus the grouped neurons with higher importance score would have larger grouped_neuron_idx

    return grouped_neurons


def get_layer_latency(lookup_table, cin=16, cout=16, k=3, fmap=128, stride=2, group_count=1, batch_size=256):
    """
        Get the latency from the lookup table
    Args:
        lookup_table (dict): the key is <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>,
            the value is the corresponding latency
        cin (int): count of input channels
        cout (int): count of output channels
        k (int): the kernel size
        fmap (int): the input feature map size
        stride (int): the stride of the conv operation
    Returns:
        lat_speed (float): the latency
    """
    if cin <= 0 or cout <= 0:
        return 0
    token_len = len(list(lookup_table.keys())[0].split('_'))
    if token_len == 6:
        key = '_'.join(
            [str(batch_size), str(int(cin)), str(int(cout)), str(fmap), str(k), str(stride)])
        if key not in lookup_table:
            near_p_count = 0
            summed_v = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    nkey = '_'.join([str(batch_size), str(int(cin+i)), str(int(cout+j)), str(fmap), str(k), str(stride)])
                    if nkey in lookup_table:
                        near_p_count += 1
                        summed_v += lookup_table[nkey]
            lat_speed = summed_v / near_p_count
        else:
            lat_speed = lookup_table[key]
    else:
        key = '_'.join(
            [str(batch_size), str(int(cin)), str(int(cout)), str(fmap), str(k), str(stride), str(int(group_count))])
        lat_speed = lookup_table[key]
    return lat_speed


def get_total_latency(layers, groups, pre_group, group_mask, fmap_table, lookup_table, lut_bs):
    """
        Get the approximated total latency for the conv network
    Args:
        layers (dict): {layer_name: the layer instance}
        groups (list): each item is a group name, which is a tuple containing all the layer names 
        pre_group (dict): {layer_name: the name of the connected former layer group}
        group_mask (dict): {group_name: the channel mask}
        fmap_table (dict): the key is the layer name, the value is the feature map size
        lookup_table (dict): the key is <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>,
            the value is the corresponding latency
    Returns:
        total_latency (float): the approximated total latency
    """
    total_latency = 0
    for group in groups:
        if group_mask is not None and group in group_mask:
            mask = group_mask[group]
            active_neuron_num = torch.sum(mask.data).cpu().item()
        else:
            active_neuron_num = layers[group[0]].weight.size(0)
        
        for layer_name in group:
            pre_group_name = pre_group[layer_name]
            if group_mask is not None and pre_group_name in group_mask:
                pre_active_neuron_num = torch.sum(
                    group_mask[pre_group_name].data).cpu().item() if pre_group_name is not None else 3
            else:
                pre_active_neuron_num = layers[pre_group_name[0]].weight.size(0) if pre_group_name is not None else 3
            layer = layers[layer_name]
            k = layer.kernel_size[0]
            fmap = fmap_table[layer_name]
            stride = layer.stride[0]
            group_count = pre_active_neuron_num if layer.groups > 1 else 1
            latency = get_layer_latency(lookup_table, pre_active_neuron_num, active_neuron_num, k, fmap, stride, group_count, batch_size=lut_bs)
            total_latency += latency
    
    return total_latency


def get_group_latency_contribute(layers, groups, pre_group, aft_group_list, group_mask, fmap_table, lookup_table, lut_bs):
    """
        Get the approximated latency reduction for grouped neurons
    Args:
        layers (dict): {layer_name: the layer instance}
        groups (list): each item is a group name, which is a tuple containing all the layer names 
        pre_group (dict): {layer_name: the name of the connected former layer group}
        aft_group_list (dict): {group_name: the name list of the connected following layer groups}
        group_mask (dict): {group_name: the channel mask}
        fmap_table (dict): the key is the layer name, the value is the feature map size
        lookup_table (dict): the key is <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>,
            the value is the corresponding latency
    Returns:
        group_latency_change (dict): the key is the layer group name, which is a tuple containing all the
            layers in the group
    """
    group_latency_change = {}
    for group in groups:
        if group_mask is not None and group in group_mask:
            mask = group_mask[group]
            active_neuron_num = torch.sum(mask.data).cpu().item()
        else:
            active_neuron_num = layers[group[0]].weight.size(0)
        latency_change = 0
        # latency change caused by the neuron num change in the pruned layers
        for layer_name in group:
            pre_group_name = pre_group[layer_name]
            if group_mask is not None and pre_group_name in group_mask:
                pre_active_neuron_num = torch.sum(
                    group_mask[pre_group_name].data).cpu().item() if pre_group_name is not None else 3
            else:
                pre_active_neuron_num = layers[pre_group_name[0]].weight.size(0) if pre_group_name is not None else 3
            layer = layers[layer_name]
            k = layer.kernel_size[0]
            fmap = fmap_table[layer_name]
            stride = layer.stride[0]
            group_count = pre_active_neuron_num if layer.groups > 1 else 1
            latency = get_layer_latency(lookup_table, pre_active_neuron_num, active_neuron_num, k, fmap, stride, group_count, batch_size=lut_bs)
            # reduced_latency = get_layer_latency(lookup_table, pre_active_neuron_num, max(active_neuron_num-CHANNEL_GROUP_SIZE, 0), k, fmap, stride)
            group_size = max([channel_group_size_dict[ln] for ln in group])
            # reduced_latency = get_layer_latency(lookup_table, pre_active_neuron_num, max(active_neuron_num-group_size, 0), k, fmap, stride)
            # layer_latency_change = latency - reduced_latency
            # latency_change += layer_latency_change

            layer_latency_change = latency / (active_neuron_num//group_size) if active_neuron_num > 0 else 0
            latency_change += layer_latency_change

        # # latency change on the following layers (the input size of the following layers will change)
        # aft_groups = aft_group_list[group]
        # for aft_group in aft_groups:
        #     if group_mask is not None and aft_group in group_mask:
        #         aft_active_neuron_num = torch.sum(group_mask[aft_group].data).cpu().item()
        #     else:
        #         aft_active_neuron_num = layers[aft_group[0]].weight.size(0)
        #     k = layers[aft_group[0]].kernel_size[0]
        #     fmap = fmap_table[aft_group[0]]
        #     stride = layers[aft_group[0]].stride[0]
        #     latency = get_layer_latency(lookup_table, active_neuron_num, aft_active_neuron_num, k, fmap, stride)
        #     reduced_latency = get_layer_latency(lookup_table, max(active_neuron_num-CHANNEL_GROUP_SIZE, 0), aft_active_neuron_num, k, fmap, stride)
        #     layer_latency_change = latency - reduced_latency
        #     latency_change += layer_latency_change
        group_latency_change[group] = latency_change
    
    return group_latency_change


def get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, group_mask, fmap_table, lookup_table, lut_bs):
    group_latency_change = {}
    for group in groups:
        if group_mask is not None and group in group_mask:
            mask = group_mask[group]
            active_neuron_num = int(torch.sum(mask.data).cpu().item())
        else:
            active_neuron_num = int(layers[group[0]].weight.size(0))
        channel_group_size = max([channel_group_size_dict[ln] for ln in group])
        channel_group_count = active_neuron_num//channel_group_size
        latency_change = [0 for _ in range(channel_group_count)]
        # latency change caused by the neuron num change in the pruned layers
        for layer_name in group:
            pre_group_name = pre_group[layer_name]
            if group_mask is not None and pre_group_name in group_mask:
                pre_active_neuron_num = torch.sum(
                    group_mask[pre_group_name].data).cpu().item() if pre_group_name is not None else 3
            else:
                pre_active_neuron_num = layers[pre_group_name[0]].weight.size(0) if pre_group_name is not None else 3
            layer = layers[layer_name]
            k = layer.kernel_size[0]
            fmap = fmap_table[layer_name]
            stride = layer.stride[0]
            conv_groups = pre_active_neuron_num if layer.groups > 1 else 1

            # get adaptive latency contribution
            for i in range(channel_group_count):
                if conv_groups == 1:
                    latency = get_layer_latency(lookup_table, pre_active_neuron_num, active_neuron_num-channel_group_size*i, k, fmap, stride, conv_groups, batch_size=lut_bs)
                    reduced_latency = get_layer_latency(lookup_table, pre_active_neuron_num, max(active_neuron_num-channel_group_size*(i+1), 0), k, fmap, stride, conv_groups, batch_size=lut_bs)
                else:
                    latency = get_layer_latency(lookup_table, pre_active_neuron_num-channel_group_size*i, active_neuron_num-channel_group_size*i, k, fmap, stride, conv_groups-channel_group_size*i, batch_size=lut_bs)
                    reduced_latency = get_layer_latency(lookup_table, max(pre_active_neuron_num-channel_group_size*(i+1), 0), max(active_neuron_num-channel_group_size*(i+1), 0), k, fmap, stride, max(conv_groups-channel_group_size*(i+1), 0), batch_size=lut_bs)
                layer_latency_change = latency - reduced_latency
                latency_change[i] += layer_latency_change
        #latency_change = [max(item, 0) for item in latency_change]

        group_latency_change[group] = latency_change

        # latency change is calculated by decreasing the output channel num,
        # thus the first calculated latency change of this layer should corresponds to the neuron(s) with least importance score.

    return group_latency_change


def knapsack_dp_adaptive(weight, value, capacity, extra_space, layer_index_split):
    assert len(weight) == len(value)
    print('++++++')
    print(capacity, sum(weight))
    if capacity >= sum(weight):
        return list(range(len(weight))), sum(weight), sum(value)

    ori_capacity = capacity
    capacity += extra_space

    weight, value, layer_index_split = weight[::-1], value[::-1], [0]+list(itertools.accumulate(layer_index_split[::-1]))
    n_items = len(value)
    # first try to reduce the complexity of the problem by finding the GCD of the weights:
    #gcd = weight[0]
    #for p in weight:
    #    gcd = math.gcd(p, gcd)
    #for i in range(len(weight)):
    #    weight[i] = weight[i] // gcd
    #capacity = capacity // gcd
    table = [[0.0] * (capacity + 1) for _ in range(2)]
    keep = [[False] * (capacity + 1) for _ in range(n_items + 1)]
    split_idx = 0
    for i in range(1, n_items + 1):
        wi = weight[i - 1]  # weight of current item
        vi = value[i - 1]  # value of current item
        index_old = (i - 1) % 2
        index_new = i % 2
        for w in range(0, capacity + 1):
            if w-wi > capacity:
                table[index_new][w] = table[index_old][w]
                continue
            val1 = vi + table[index_old][w - wi]
            val2 = table[index_old][w]
            if layer_index_split[split_idx] == i - 1:
                table[index_new][w] = val1
                keep[i][w] = True
            if (wi <= w) and (val1 > val2 or (val1==val2 and wi == 0)) and (keep[i-1][w-wi] or layer_index_split[split_idx]==i-1):
                table[index_new][w] = val1
                keep[i][w] = True
            else:
                table[index_new][w] = val2
        if i-1 == layer_index_split[split_idx]:
            split_idx += 1
    items_in_bag = []
    K = ori_capacity
    for i in range(n_items, 0, -1):
        if keep[i][K] == True:
            items_in_bag.append(n_items - 1 - (i - 1))
            K -= weight[i - 1]
    used_capacity = ori_capacity - K
    achieved_value = table[n_items % 2][ori_capacity]
    print(layer_index_split)
    print(weight)
    print(value)
    print(capacity, sum(weight))
    print('*****')
    print(items_in_bag, used_capacity, achieved_value)
    print('*****')

    return items_in_bag, used_capacity, achieved_value



def knapsack_pulp_layerprune(weight, value, capacity, extra_space, layer_index_split, results_dir):
    assert len(weight) == len(value)
    print('++++++')
    print(capacity, sum(weight))
    if capacity >= sum(weight):
        return list(range(len(weight))), sum(weight), sum(value)

    ori_capacity = capacity
    capacity += extra_space

    # weight, value, layer_index_split = weight[::-1], value[::-1], [0]+list(itertools.accumulate(layer_index_split[::-1]))
    weight, value, layer_index_split = weight, value, list(itertools.accumulate(layer_index_split))

    print(value)
    print(layer_index_split)
    n_items = len(value)

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    for layer_idx in range(len(layer_index_split)):
        pre_total_num = layer_index_split[layer_idx - 1] if layer_idx > 0 else 0
        layer_value = value[pre_total_num : layer_index_split[layer_idx]]
        sorted_value = sorted(layer_value)
        if layer_value != sorted_value:
            raise ValueError(
                "The importance of neuron groups from the same layer should be sorted ascendingly."
            )
    problem = LpProblem(name="resource-allocation", sense=LpMaximize)
    # Define the decision variables
    # Each variable is a binary value that 1 means being selected to keep while 0 means to prune.
    decision_var = [LpVariable(name=f"y{i}", cat=LpBinary) for i in range(n_items)]

    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. Preceding constraint: for neurons in the same layer, the more important one always
    # needs to be kept before a less important one'
    # 3. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    problem += (
        lpSum([decision_var[idx] * weight[idx] for idx in range(n_items)]) <= capacity,
        "latency_constraint",
    )
    split_idx = 0
    print(len(decision_var))
    print(n_items)
    print(layer_index_split)
    for idx in range(1, n_items + 1):
        if idx != layer_index_split[split_idx]:
            problem += (
                decision_var[idx - 1] <= decision_var[idx],
                f"preceding_constraint{idx}",
            )
        else:
            split_idx += 1

    # Set objective
    total_importance = 0
    for idx, importance in enumerate(value):
        total_importance += decision_var[idx] * importance
    problem += total_importance

    if results_dir is not None:
        # Save the formed problem for debug purpose
        problem.writeMPS(f"{results_dir}/problem.mps")

    # Set random seed to the solver to ensure reproducability.
    # See issue https://github.com/coin-or/pulp/issues/545
    # cbc_solver = PULP_CBC_CMD(options=["RandomS 20"])
    cbc_solver = PULP_CBC_CMD(options=["RandomS 0"])
    problem.solve(cbc_solver)

    # Check the status
    if LpStatus[problem.status] != "Optimal":
        logger.warning(
            "Lead to a {} solution with capacity {}.".format(
                LpStatus[problem.status], capacity
            )
        )

    items_in_bag = []
    for var in decision_var:
        # For the variable being set to 1, the corresponding neuron group is selected to keep.
        if var.value() == 1:
            items_in_bag.append(int(var.name.replace("y", "")))    

    print(items_in_bag)

    return items_in_bag, None, None


def knapsack_pulp(weight, value, capacity, extra_space, layer_index_split, results_dir):
    assert len(weight) == len(value)
    print('++++++')
    print(capacity, sum(weight))
    if capacity >= sum(weight):
        return list(range(len(weight))), sum(weight), sum(value)

    ori_capacity = capacity
    capacity += extra_space

    # weight, value, layer_index_split = weight[::-1], value[::-1], [0]+list(itertools.accumulate(layer_index_split[::-1]))
    weight, value, layer_index_split = weight, value, list(itertools.accumulate(layer_index_split))

    print(value)
    print(layer_index_split)
    n_items = len(value)

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    for layer_idx in range(len(layer_index_split)):
        pre_total_num = layer_index_split[layer_idx - 1] if layer_idx > 0 else 0
        layer_value = value[pre_total_num : layer_index_split[layer_idx]]
        sorted_value = sorted(layer_value)
        if layer_value != sorted_value:
            raise ValueError(
                "The importance of neuron groups from the same layer should be sorted ascendingly."
            )
    problem = LpProblem(name="resource-allocation", sense=LpMaximize)
    # Define the decision variables
    # Each variable is a binary value that 1 means being selected to keep while 0 means to prune.
    decision_var = [LpVariable(name=f"y{i}", cat=LpBinary) for i in range(n_items)]

    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. Preceding constraint: for neurons in the same layer, the more important one always
    # needs to be kept before a less important one'
    # 3. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    problem += (
        lpSum([decision_var[idx] * weight[idx] for idx in range(n_items)]) <= capacity,
        "latency_constraint",
    )
    split_idx = 0
    print(len(decision_var))
    print(n_items)
    print(layer_index_split)
    for idx in range(1, n_items + 1):
        if idx != layer_index_split[split_idx]:
            problem += (
                decision_var[idx - 1] <= decision_var[idx],
                f"preceding_constraint{idx}",
            )
        else:
            problem += (
                decision_var[idx - 1] >= 1,
                f"no_layer_prune_constraint{split_idx}",
            )
            split_idx += 1

    # Set objective
    total_importance = 0
    for idx, importance in enumerate(value):
        total_importance += decision_var[idx] * importance
    problem += total_importance

    if results_dir is not None:
        # Save the formed problem for debug purpose
        problem.writeMPS(f"{results_dir}/problem.mps")

    # Set random seed to the solver to ensure reproducability.
    # See issue https://github.com/coin-or/pulp/issues/545
    # cbc_solver = PULP_CBC_CMD(options=["RandomS 20"])
    cbc_solver = PULP_CBC_CMD(options=["RandomS 0"])
    problem.solve(cbc_solver)

    # Check the status
    if LpStatus[problem.status] != "Optimal":
        print(
            "Lead to a {} solution with capacity {}.".format(
                LpStatus[problem.status], capacity
            )
        )

    items_in_bag = []
    for var in decision_var:
        # For the variable being set to 1, the corresponding neuron group is selected to keep.
        if var.value() == 1:
            items_in_bag.append(int(var.name.replace("y", "")))    

    print(items_in_bag)

    return items_in_bag, None, None



def knapsack_pyomo(weight, value, capacity, extra_space, layer_index_split, results_dir):
    print("Running Pyomo")
    assert len(weight) == len(value)
    print('++++++')
    print(capacity, sum(weight))
    if capacity >= sum(weight):
        return list(range(len(weight))), sum(weight), sum(value)

    ori_capacity = capacity
    capacity += extra_space

    # weight, value, layer_index_split = weight[::-1], value[::-1], [0]+list(itertools.accumulate(layer_index_split[::-1]))
    weight, value, layer_index_split = weight, value, list(itertools.accumulate(layer_index_split))

    print(value)
    print(layer_index_split)
    n_items = len(value)

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    for layer_idx in range(len(layer_index_split)):
        pre_total_num = layer_index_split[layer_idx - 1] if layer_idx > 0 else 0
        layer_value = value[pre_total_num : layer_index_split[layer_idx]]
        sorted_value = sorted(layer_value)
        if layer_value != sorted_value:
            raise ValueError(
                "The importance of neuron groups from the same layer should be sorted ascendingly."
            )
    all_items = list(range(n_items))
    model = ConcreteModel()
    model.decision_var = Var(all_items, domain=Binary)

    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. Preceding constraint: for neurons in the same layer, the more important one always
    # needs to be kept before a less important one'
    # 3. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    model.latency_constraint = Constraint(expr=sum(weight[i] * model.decision_var[i] for i in all_items) <= capacity)
    model.preceding_constraint = ConstraintList()
    model.no_layer_prune_constraint = ConstraintList()

    split_idx = 0
    for idx in range(1, n_items + 1):
        if idx != layer_index_split[split_idx]:
            model.preceding_constraint.add(model.decision_var[idx - 1] <= model.decision_var[idx])
        else:
            model.no_layer_prune_constraint.add(model.decision_var[idx - 1] >= 1)
            split_idx += 1

    
    # Set objective
    model.obj = Objective(expr=sum(value[i] * model.decision_var[i] for i in all_items), sense=maximize)

    # Solve!
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    print("Objective value:", model.obj())
    print("Status = %s" % results.solver.termination_condition)

    items_in_bag = []
    for i in all_items:
        if model.decision_var[i].value == 1:
            items_in_bag.append(i) 

    print(items_in_bag)
    return items_in_bag, None, None


def knapsack(weight, value, capacity, wrap_solver=None):
    """
        Solve 0-1 knapsack problem. We are trying to solve the problem that given a set of neurons, select the neurons that
        achieve the maximum total importance value under the constraint of latency limit.

        The knapsack problem is solved using Google Optimization Tools (a.k.a., OR-Tools). 
        More information can be found at https://github.com/google/or-tools. 

        Note: There will be unknown error if we import the or-tools after apex imported. So we import or-tools at the very 
        beginning of the program and pass the imported solver as an arg. 
    Args:
        weight (list[int]): the weight of the items, in our case, is the latency 
        value (list[int]): the value of the items, in our case, is the importance
        capacity (int): the capacity of the knapsack, in our case, is the latency limit
    Returns:
        items_in_bag (list[int]): list of item's index that is in the bag to achieve maximum value
        used_capacity (int): the total weights used to achive the best value.
        achieved_value (int): the largest total value computed. 
    """
    assert len(weight) == len(value)

    # Create the solver.
    solver = wrap_solver.KnapsackSolver(
        wrap_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'Knapsack')

    solver.Init(value, [weight], [capacity])
    achieved_value = solver.Solve()
    
    items_in_bag = []
    used_capacity = 0
    for i in range(len(value)):
        if solver.BestSolutionContains(i):
            items_in_bag.append(i)
            used_capacity += weight[i]

    return items_in_bag, used_capacity, achieved_value


def prune_according_to_rank(weight, value, capacity):
    """
        Select neurons according to importance rank, until the targeted latency reached.
    Args:
        weight (list[int/float]): the weight of the items, in our case, is the latency 
        value (list[int/float]): the value of the items, in our case, is the importance
        capacity (int/float): the capacity of the knapsack, in our case, is the latency limit
    Returns:
        items_in_bag (list[int]): list of item's index that is in the bag to achieve maximum value
        used_capacity (int/float): the total weights used to achive the best value.
    """
    assert len(weight) == len(value)

    sorted_args = np.argsort(np.array(value))[::-1]
    sorted_weight = np.array(weight)[sorted_args]
    cumsum_weight = np.cumsum(sorted_weight)

    if cumsum_weight[-1] <= capacity:
        max_idx = len(cumsum_weight)
    else:
        max_idx = np.where(cumsum_weight > capacity)[0][0]

    items_in_bag = sorted_args[:max_idx]
    used_capacity = cumsum_weight[max_idx-1]
    
    return items_in_bag, used_capacity


def apply_latency_target_pruning(output_dir, layers, layer_bn, blocks, layer_gate, metric, target_latency, group_mask,
                                 groups, pre_group, aft_group_list, fmap_table, lookup_table, method, 
                                 wrap_solver=None, mu=0., step_size=-1, lut_bs=256, pulp=False, pyomo=False):
    """
        Apply latency targeted pruning to ensure the pruned network under a certain latency constraint.
    Args:
        layers (dict): {layer_name: the layer instance}
        layer_bn (dict): {layer_name: the corrsponding following bn layer name}
        layer_gate (dict): {layer_name: the corresponding following gate layer name}
        metric (dict): the metric to measure the importance of the neurons, {group_name: tensor of neuron importance}
        target_latency (float): the targeted latency
        group_mask (dict): {group_name: the channel mask}
        groups (list): each item is a group name, which is a tuple containing all the layer names 
        pre_group (dict): {layer_name: the name of the connected former layer group}
        aft_group_list (dict): {group_name: the name list of the connected following layer groups}       
        fmap_table (dict): the key is the layer name, the value is the feature map size
        lookup_table (dict): the key is <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>,
            the value is the corresponding latency
        method (int): the method going to use for pruning.
        wrap_solver: the solver imported from or-tools
        mu (float): the scalar factor for latency aware importance
    Returns:
        total_pruned_num (int): the total number of neurons being pruned. 
    """
    global prune_counter
    if step_size > 0:
        for k in channel_group_size_dict.keys():
            channel_group_size_dict[k] = min(step_size, int(layers[k].weight.size(0)))
    
    # adaptive latency change
    # {grouped_neuron_idx: {"layer_group_name": name, "channel_indices": [idx], "combined_importance": value}}
    # in each layer, the neuron group with higher importance score has larger grouped_neuron_idx
    grouped_neurons = group_neurons_by_rank(metric, group_mask)
    # {group_name: [latency_change]}
    # in each layer, the list of latency change is calculated by decreasing the out channel gradually, adaptively
    
    # every element is a 2d matrix looping through previous and current number of channels
    adaptive_group_latency_change = get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, group_mask, fmap_table, lookup_table, lut_bs)
    grouped_neuron_idx_list = range(len(list(grouped_neurons.keys())))
    print("Total number of neurons")
    print(len(grouped_neuron_idx_list))
    importance_list = [round(grouped_neurons[idx]["combined_importance"]*1000000) for idx in grouped_neuron_idx_list]
    # latency_list = [int(item*100) for idx in grouped_neuron_idx_list for item in adaptive_group_latency_change[grouped_neurons[idx]["layer_group_name"]]]
    layer_index_split = []

    cur_gn, cnt = grouped_neurons[0]['layer_group_name'], 0
    latency_list = [round(item*1000) for item in adaptive_group_latency_change[cur_gn]]
    for idx in grouped_neuron_idx_list:
        if grouped_neurons[idx]['layer_group_name'] == cur_gn:
            cnt += 1
        else:
            layer_index_split.append(cnt)
            cur_gn, cnt = grouped_neurons[idx]['layer_group_name'], 1
            latency_list.extend([round(item*1000) for item in adaptive_group_latency_change[cur_gn]])
    layer_index_split.append(cnt)
    assert sum(layer_index_split) == len(list(grouped_neurons.keys()))
    
    # with open(f"layersplit_time{prune_counter}.pkl", 'wb') as f:
    #     pickle.dump(layer_index_split, f)

    neg_latency = [item for item in latency_list if item < 0]
    extra_space = 0 if len(neg_latency)==0 else abs(sum(neg_latency))

    total_latency = get_total_latency(layers, groups, pre_group, group_mask, fmap_table, lookup_table, lut_bs)
    prev_latency = total_latency
    summed_latency = sum([sum(adaptive_group_latency_change[group_name]) for group_name in adaptive_group_latency_change.keys()])
    print(f"total_latency:{total_latency}, summed_latency_contribute:{summed_latency}")
    print(f"neg space: {extra_space}")

    if pulp:
        print("Running pulp solver for knapsack")
        # items_in_bag_layerpruned, _, _ = knapsack_pulp_layerprune(latency_list, importance_list, round(target_latency*1000), extra_space, layer_index_split, output_dir)
        items_in_bag, _, _ = knapsack_pulp(latency_list, importance_list, round(target_latency*1000), extra_space, layer_index_split, output_dir)
        # should_pruned = set(items_in_bag) - set(items_in_bag_layerpruned)
        # with open(f"should_pruned_time{prune_counter}.pkl", 'wb') as f:
        #     pickle.dump(should_pruned, f)
    elif pyomo:
        print("Running pyomo solver for knapsack")
        items_in_bag, _, _ = knapsack_pyomo(latency_list, importance_list, round(target_latency*1000), extra_space, layer_index_split, output_dir)
    else:
        items_in_bag, used_capacity, achieved_value = knapsack_dp_adaptive(latency_list, importance_list, round(target_latency*1000), extra_space, layer_index_split)

        # used_capacity /= 1000

    
    pruned_items = set(grouped_neuron_idx_list) - set(items_in_bag)
    pruned_dict = {}
    for pruned_idx in list(pruned_items):
        layer_group_name = grouped_neurons[pruned_idx]["layer_group_name"]
        pruned_channel_sum = len(grouped_neurons[pruned_idx]["channel_indices"])
        for layer_name in layer_group_name:
            if layer_name not in pruned_dict:
                pruned_dict[layer_name] = 0
            pruned_dict[layer_name] += pruned_channel_sum
    print("Pruned Distribution", pruned_dict)
    with open(f"pruned_dict_halp{prune_counter}.pkl", 'wb') as f:
        pickle.dump(pruned_dict, f)
    prune_counter += 1
    # reset the mask to 0
    ori_channel_num = {}
    for group_name in metric.keys():
        ori_channel_num[group_name] = int(torch.sum(group_mask[group_name]).item())
        group_mask[group_name][:] = 0.

    # set the selected neurons to active 
    for item_idx in items_in_bag:
        layer_group_name = grouped_neurons[item_idx]["layer_group_name"]
        channel_indices = grouped_neurons[item_idx]["channel_indices"]
        group_mask[layer_group_name][channel_indices] = 1.

    total_pruned_num = 0
    for group_name, mask in group_mask.items():
        cur_channel_num = int(torch.sum(mask).item())
        pruned_channel = ori_channel_num[group_name] - cur_channel_num
        total_pruned_num += pruned_channel*len(group_name)
        # print('*** Group {}: {} channels / {} neurons are pruned at current step. '
        #       '{} channels / {} neurons left. ***'.format(group_name,
        #                                                   pruned_channel,
        #                                                   pruned_channel*len(group_name),
        #                                                   cur_channel_num,
        #                                                   cur_channel_num*len(group_name)))

    # mask conv and bn parameters 
    mask_conv_bn(layers, layer_bn, layer_gate, group_mask)

    # get the total latency after pruning
    total_latency = get_total_latency(layers, groups, pre_group, group_mask, fmap_table, lookup_table, lut_bs)
    print(f"Prev Latency after pruning: {prev_latency}")
    print(f"Actual Latency after pruning: {total_latency}")

    # print('Achieved latency: {}, actual latency after pruning: {}'.format(used_capacity, total_latency))

    return total_pruned_num
    

def mask_conv_bn(layers, layer_bn, layer_gate, group_mask):
    """
        Mask the pruned neurons, set the corresponding neurons to zero
    Args:
        layers (dict): {layer_name: the layer instance}
        layer_bn (dict): {layer_name: the corrsponding following bn layer name}
        layer_gate (dict): {layer_name: the corresponding following gate layer name}
        group_mask (dict): {group_name: the channel mask}
    """
    for group_name, mask in group_mask.items():
        if layer_gate is not None:
            gate_layer = layers[layer_gate[group_name[0]]]
            gate_layer.weight.data.mul_(mask)
        for layer_name in group_name:
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
