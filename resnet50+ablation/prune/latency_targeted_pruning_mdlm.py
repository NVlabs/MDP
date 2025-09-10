import torch
import math
import itertools
import numpy as np
from collections import defaultdict
from utils.utils import count_non_zero_neurons
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
# from pyomo.core.base.expr import identify_variables
import pickle
import os
import time
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
prune_counter = 0

EPS = 1e-10
channel_group_size_dict = {}

def load_group_size(arch):
    global channel_group_size_dict
    with open('group_size/{}_group_size.json'.format(arch), 'r') as f:
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
        # Skip tricks
#         if 'module.conv1' in group or ('module.features.conv_bn.conv' in group and int(value_remained.size(0)) <= 16):
# #         if 'module.conv1' in group:
#             combined_importance[:] = 10000  # set to large values to avoid pruning
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

# Done Change for latency2
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

            if layer_name == "module.conv1" or layer_name == 'module.features.conv_bn.conv':
                latency_change = np.zeros((channel_group_count + 1))
                for i in range(channel_group_count + 1):
                    latency = get_layer_latency(lookup_table, pre_active_neuron_num, channel_group_size*i, k, fmap, stride, conv_groups, batch_size=lut_bs)
                    latency_change[i] = round(latency*1000)
            else:
                pre_channel_group_size = max([channel_group_size_dict[ln] for ln in pre_group_name])
                pre_channel_group_count = pre_active_neuron_num//pre_channel_group_size
                pre_channel_group_count = int(pre_channel_group_count)

                # ************************************************************************************** #
                # Handle DW-Conv for MobileNet
                if conv_groups > 1:
                    assert pre_active_neuron_num == active_neuron_num 
                    latency_change = np.zeros((channel_group_count + 1))
                    for i in range(channel_group_count + 1):
                        latency = get_layer_latency(lookup_table, channel_group_size*i, channel_group_size*i, k, fmap, stride, channel_group_size*i, batch_size=lut_bs)
                        latency_change[i] = round(latency*1000)

                else:
                    # print(channel_group_count, pre_channel_group_count)
                    latency_change = np.zeros((channel_group_count + 1, pre_channel_group_count + 1))
                    for i in range(channel_group_count + 1):
                        for j in range(pre_channel_group_count + 1):
                            latency = get_layer_latency(lookup_table, pre_channel_group_size*j, channel_group_size*i, k, fmap, stride, conv_groups, batch_size=lut_bs)
                            latency_change[i, j] = round(latency*1000)
                
            key_name = layer_name
            group_latency_change[key_name] = (pre_group_name, latency_change)
    
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



def knapsack_pyomo(weight_dict, value_dict, capacity, extra_space, layer_index_split, results_dir, init_values=None):
    print("Running Pyomo")
    print('++++++')

    ori_capacity = capacity
    capacity += extra_space

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    model = ConcreteModel()
    group_var_slices = {}
    counter = 0
    for group_name, value in value_dict.items():
        # print(group_name)
        group_var_slices[group_name] = (counter, counter+len(value))
        counter += len(value)
    all_items = list(range(counter))
    # model.decision_vars = Var(all_items, domain=Binary)
    if init_values is not None:    
        model.decision_vars = Var(all_items, domain=Binary, initialize=init_values)
    else:
        model.decision_vars = Var(all_items, domain=Binary)
    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    # 3. Only one unique configuration for one group
    latency_expr = 0
    importance = 0
    model.no_layer_prune_constraint = ConstraintList()
    model.group_unique_constraint = ConstraintList()
    for group_name, value in value_dict.items():
        cur_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[group_name][0], group_var_slices[group_name][1])]
#         model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        # The following two are the same functionality. For mobilenet with smaller number of data, upper row seems to give
        # optimizer larger freedom to solve. Not important though.
        model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        # model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
        model.no_layer_prune_constraint.add(cur_decision_vars[0] <= 0)
        for layer_name in group_name:
            if layer_name == "module.conv1":
                pre_group_name, lat_vec = weight_dict[layer_name]
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                # model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] == 1)
                model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] >= 1)
                continue
            # ************************************************************************************** #
            # Handle DW-Conv for MobileNet
            if ("conv1" in layer_name and "module.features" in layer_name) or layer_name == "module.features.conv_bn.conv":
                pre_group_name, lat_vec = weight_dict[layer_name]
#                 print(lat_vec.shape)
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                continue
            pre_group_name, lat_matrix = weight_dict[layer_name]
            # pre_decision_vars = model.decision_vars[group_var_slices[pre_group_name][0]:group_var_slices[pre_group_name][1]]
            pre_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[pre_group_name][0], group_var_slices[pre_group_name][1])]
            # lat_matrix: 1st dimension current group number, 2nd dimension previous group
            # channel_group_count x pre_channel_group_count
            # cur_decision_vars x (lat_matrix x pre_decision_vars.T)
            cur_expr = 0
            assert lat_matrix.shape[0] == len(cur_decision_vars)
            for i in range(len(cur_decision_vars)):
                cur_expr += cur_decision_vars[i] * sum(lat_matrix[i, j] * pre_decision_vars[j] for j in range(len(pre_decision_vars)))
            latency_expr += cur_expr
        importance += sum(value[i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))

    model.latency_constraint = Constraint(expr=latency_expr <= capacity)
    
    # Set objective
    model.obj = Objective(expr=importance, sense=maximize)
    # model.obj = Objective(expr=sum(model.decision_vars), sense=maximize)
    # model.obj = Objective(expr=sum(model.decision_vars[i]*1 for i in all_items), sense=maximize)

    # Solve!
    # Mixed-Integer Nonlinear
    # model.obj.display()
    # model.display()
    # model.pprint()
    start = time.time()
    
    solver = SolverFactory('mindtpy')
    # solver = SolverFactory('glpk')
    # Without FP, problem is not solvable
    # if tries % 2 == 0:
    #     results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt')
    # else:
    #     results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt')

    results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    # Initializing variable manually and search with FP is actually better for our usecase
    # results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    
    end = time.time()
    # results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    # results = solver.solve(model) 
    print("Objective value:", model.obj())
    # print("Status = %s" % results.termination_condition)
    num_channel_to_keep = {}
    items_in_bag = []
    # Post-process the results
    slices = {}
    counter = 0
    for group_name, value in value_dict.items():
        # print(group_name)
        slices[group_name] = (counter, counter+len(value)-1)
        counter += (len(value)-1)

    for group_name, value in value_dict.items():
        # print(group_name)
        indices = list(range(group_var_slices[group_name][0], group_var_slices[group_name][1]))
        cur_decision_vars = [model.decision_vars[k] for k in indices]
        cur_slices = list(range(slices[group_name][0], slices[group_name][1]))
        for i in range(len(cur_decision_vars)):
            # print(cur_decision_vars[i].value)
            # How many groups to keep?
            if cur_decision_vars[i].value == 1:
                if i == 0:
                    print(f"No channels for group {group_name}")
                    break
                num_channel_to_keep[group_name] = i
                # We get the bottom indices because grouped_neurons are sorted ascendingly, and we want the highest-valued groups
                keep_idxs = cur_slices[-i:]
                # print(len(keep_idxs), i)
                # print(keep_idxs)
                items_in_bag += keep_idxs
    print("Solve Time", end-start)
    print("Finished")
    # exit()
    # for group_name, value in value_dict.items():
    #     print(group_name)
    #     print(model.decision_vars[group_var_slices[group_name][0]:group_var_slices[group_name][1]])
    # print(items_in_bag)
    return items_in_bag, None, None

def knapsack_pyomo_mobilenet(weight_dict, value_dict, capacity, extra_space, layer_index_split, results_dir, init_values=None):
    print("Running Pyomo")
    print('++++++')

    ori_capacity = capacity
    capacity += extra_space

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    model = ConcreteModel()
    group_var_slices = {}
    counter = 0
    for group_name, value in value_dict.items():
        # print(group_name)
        group_var_slices[group_name] = (counter, counter+len(value))
        counter += len(value)
    all_items = list(range(counter))
    # model.decision_vars = Var(all_items, domain=Binary)
    if init_values is not None:    
        model.decision_vars = Var(all_items, domain=Binary, initialize=init_values)
    else:
        model.decision_vars = Var(all_items, domain=Binary)
    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    # 3. Only one unique configuration for one group
    latency_expr = 0
    importance = 0
    model.no_layer_prune_constraint = ConstraintList()
    model.group_unique_constraint = ConstraintList()
    for group_name, value in value_dict.items():
        cur_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[group_name][0], group_var_slices[group_name][1])]
#         model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        # The following two are the same functionality. For mobilenet with smaller number of data, upper row seems to give
        # optimizer larger freedom to solve. Not important though.
        model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
        # model.no_layer_prune_constraint.add(cur_decision_vars[0] <= 0)
        for layer_name in group_name:
            # minimum 16 for mobilenet
            if layer_name == "module.features.conv_bn.conv":
                # group size is 2
                for i in range(1, 8):
                    model.no_layer_prune_constraint.add(cur_decision_vars[i] == 0)

            if layer_name == "module.conv1":
                pre_group_name, lat_vec = weight_dict[layer_name]
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] == 1)
                # model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] >= 1)
                continue
            # ************************************************************************************** #
            # Handle DW-Conv for MobileNet
            if ("conv1" in layer_name and "module.features" in layer_name) or layer_name == "module.features.conv_bn.conv":
                pre_group_name, lat_vec = weight_dict[layer_name]
#                 print(lat_vec.shape)
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                continue
            pre_group_name, lat_matrix = weight_dict[layer_name]
            # pre_decision_vars = model.decision_vars[group_var_slices[pre_group_name][0]:group_var_slices[pre_group_name][1]]
            pre_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[pre_group_name][0], group_var_slices[pre_group_name][1])]
            # lat_matrix: 1st dimension current group number, 2nd dimension previous group
            # channel_group_count x pre_channel_group_count
            # cur_decision_vars x (lat_matrix x pre_decision_vars.T)
            cur_expr = 0
            assert lat_matrix.shape[0] == len(cur_decision_vars)
            for i in range(len(cur_decision_vars)):
                cur_expr += cur_decision_vars[i] * sum(lat_matrix[i, j] * pre_decision_vars[j] for j in range(len(pre_decision_vars)))
            latency_expr += cur_expr
        importance += sum(value[i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))

    model.latency_constraint = Constraint(expr=latency_expr <= capacity)
    
    # Set objective
    model.obj = Objective(expr=importance, sense=maximize)
    # model.obj = Objective(expr=sum(model.decision_vars), sense=maximize)
    # model.obj = Objective(expr=sum(model.decision_vars[i]*1 for i in all_items), sense=maximize)

    # Solve!
    # Mixed-Integer Nonlinear
    # model.obj.display()
    # model.display()
    # model.pprint()
    start = time.time()
    
    solver = SolverFactory('mindtpy')
    # solver = SolverFactory('glpk')
    # Without FP, problem is not solvable
    # if tries % 2 == 0:
    #     results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt')
    # else:
    #     results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt')

    results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    # Initializing variable manually and search with FP is actually better for our usecase
    # results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    
    end = time.time()
    # results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    # results = solver.solve(model) 
    print("Objective value:", model.obj())
    # print("Status = %s" % results.termination_condition)
    num_channel_to_keep = {}
    items_in_bag = []
    # Post-process the results
    slices = {}
    counter = 0
    for group_name, value in value_dict.items():
        # print(group_name)
        slices[group_name] = (counter, counter+len(value)-1)
        counter += (len(value)-1)

    for group_name, value in value_dict.items():
        # print(group_name)
        indices = list(range(group_var_slices[group_name][0], group_var_slices[group_name][1]))
        cur_decision_vars = [model.decision_vars[k] for k in indices]
        cur_slices = list(range(slices[group_name][0], slices[group_name][1]))
        for i in range(len(cur_decision_vars)):
            # print(cur_decision_vars[i].value)
            # How many groups to keep?
            if cur_decision_vars[i].value == 1:
                if i == 0:
                    print(f"No channels for group {group_name}")
                    break
                num_channel_to_keep[group_name] = i
                # We get the bottom indices because grouped_neurons are sorted ascendingly, and we want the highest-valued groups
                keep_idxs = cur_slices[-i:]
                # print(len(keep_idxs), i)
                # print(keep_idxs)
                items_in_bag += keep_idxs
    print("Solve Time", end-start)
    print("Finished")
    # exit()
    # for group_name, value in value_dict.items():
    #     print(group_name)
    #     print(model.decision_vars[group_var_slices[group_name][0]:group_var_slices[group_name][1]])
    # print(items_in_bag)
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

    # group_neurons_inputs = [metric, group_mask]
    # with open(f"{prune_counter}_group_inputs_latv2.pkl", 'wb') as f:
    #     pickle.dump(group_neurons_inputs, f)
    # adaptive latency change
    # {grouped_neuron_idx: {"layer_group_name": name, "channel_indices": [idx], "combined_importance": value}}
    # in each layer, the neuron group with higher importance score has larger grouped_neuron_idx
    grouped_neurons = group_neurons_by_rank(metric, group_mask)
    # {group_name: [latency_change]}
    # in each layer, the list of latency change is calculated by decreasing the out channel gradually, adaptively
    # if not os.path.isfile("latency_dict.pkl"):
    adaptive_group_latency_change = get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, group_mask, fmap_table, lookup_table, lut_bs)
    latency_dict = adaptive_group_latency_change
    print("Got the adaptive_group_latency_change")
    grouped_neuron_idx_list = range(len(list(grouped_neurons.keys())))
    importance_dict = {}
    # Group Neurons are in ascending order
    for idx in grouped_neuron_idx_list:
        group_name = grouped_neurons[idx]["layer_group_name"]
        cur_list = importance_dict.get(group_name, [])
        cur_list.append(round(grouped_neurons[idx]["combined_importance"]*1000000))
        importance_dict[group_name] = cur_list
    
    for key, val in importance_dict.items():
        print(key, val)

    for key in importance_dict:
        importance_dict[key] = list(itertools.accumulate(importance_dict[key][::-1]))
        # None selected for this layer
        importance_dict[key] = [0] + importance_dict[key]
    for key, val in importance_dict.items():
        print(key, val)

    # neg_latency = [item for item in latency_list if item < 0]
    # extra_space = 0 if len(neg_latency)==0 else abs(sum(neg_latency))

    total_latency = get_total_latency(layers, groups, pre_group, group_mask, fmap_table, lookup_table, lut_bs)
    prev_latency = total_latency
    # summed_latency = sum([sum(adaptive_group_latency_change[group_name]) for group_name in adaptive_group_latency_change.keys()])
    print(f"total_latency:{total_latency}")
    # print(f"neg space: {extra_space}")

    extra_space = 0
    layer_index_split = None
    print("Running pyomo solver for knapsack")
    # if os.path.isfile("latency_dict.pkl"):
    #     with open("latency_dict.pkl", "rb") as f:
    #          latency_dict = pickle.load(f)
    # else:
    #     with open("latency_dict.pkl", 'wb') as f:
    #         pickle.dump(latency_dict, f)
    
    inputs = [latency_dict, importance_dict, round(target_latency*1000), extra_space, layer_index_split, output_dir]
    with open("solver_inputs.pkl", 'wb') as f:
        pickle.dump(inputs, f)
    
    # We need to scale down for easier pyomo usecase
    # for key, val in importance_dict.items():
    #     importance_dict[key] = [float(x/1000000.0) for x in val]
    # for key, val in importance_dict.items():
    #     if key == ('module.conv1',):
    #         print("Scale down first layer")
    #         importance_dict[key] = [x//100000000 for x in val]
    
    inputs = [latency_dict, importance_dict, round(target_latency*1000), extra_space, layer_index_split, output_dir]
    with open(f"{prune_counter}_solver_inputs_latv2_2nd.pkl", 'wb') as f:
        pickle.dump(inputs, f)

    tries = 0
    items_in_bag, _, _ = knapsack_pyomo(latency_dict, importance_dict, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=None)
    # items_in_bag, _, _ = knapsack_pyomo_mobilenet(latency_dict, importance_dict, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=None)
    # while tries < 6:
    #     try:
    #         if tries == 1:
    #             init_values = 0
    #         elif tries == 2:
    #             init_values = 1
    #         else:
    #             init_values = None
            
    #         if tries >= 3:
    #             for key, val in importance_dict.items():
    #                 print("Scaling down for easier handling")
    #                 importance_dict[key] = [x//1000 for x in val]

    #         items_in_bag, _, _ = knapsack_pyomo(latency_dict, importance_dict, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=init_values)
    #         break
    #     except:
    #         print("error. trying again...")
    #         tries += 1
    #         if tries == 5:
    #         # if tries == 1:
    #             print("Not pruning this step")
    #             items_in_bag = grouped_neuron_idx_list
    #             break
        # used_capacity /= 1000
    # print(grouped_neuron_idx_list)
    # reset the mask to 0
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
    with open(f"pruned_dict_latv2{prune_counter}_2nd.pkl", 'wb') as f:
        pickle.dump(pruned_dict, f)
    prune_counter += 1

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
