import torch
import math
import itertools
import numpy as np
from collections import defaultdict
import pickle
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
from prune.hahp_func import *
import time
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
used_keys = []
fmap_mapping = {
    224: 256,
    56: 64,
    28: 32,
    14: 16,
    7: 8
}

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


def group_neurons_by_rank(metric, layer_masks, layers, groups, blocks):
    """
        Group the neurons/channels with 8 neurons/channels per group to make the pruned structure GPU friendly.
    Args:
        metric (dict): the metric to measure the importance of the neurons
        group_mask (dict): the mask for each layer group indicating which channels have been zeroed.
    Returns:
        grouped_neruons (dict): indicates the layer group name, the channel index and the total importance score 
            of each neuron group.
    """
    grouped_neurons = []  # {grouped_neuron_idx: {"layer_group_name": name, "channel_indices": [idx], "combined_importance": value}}

    grouped_neuron_idx = 0
    block_list = sorted(blocks.keys())
    print(block_list)
    used_layers = []
    for group in sorted(groups):
        # Calculate group combined importance
        importance_value = 0
        for layer_name in group:
            importance_value += metric[layer_name]
        _, mask = get_mask_for_group(group, layers, layer_masks)
        
        if mask is not None:
            value_remained = importance_value[mask == 1.]
            channels_remained = np.arange(mask.size(0))[mask.cpu().numpy() == 1.]
        else:
            for layer_name in group:
                layer_masks[layer_name] = torch.ones_like(importance_value)
            value_remained = importance_value
            channels_remained = np.arange(layer_masks[group[0]].size(0))

        group_size = max([channel_group_size_dict[ln] for ln in group])
        assert 'module.conv1' in group or int(value_remained.size(0)) % max([channel_group_size_dict[ln] for ln in group]) == 0
        group_sorted_values, group_sorted_indices = torch.sort(value_remained)
        group_sorted_indice2 = group_sorted_indices.view(-1, group_size)
        # group_sorted_values2 = group_sorted_values.view(-1, group_size)
        # group_combined_importance = group_sorted_values2.sum(dim=1)
        # print(group, group_combined_importance)
        # print("group", group)
        for layer_name in group:
            active_neuron, layer_mask = get_mask_for_layer(layer_name, layers, layer_masks)
            if active_neuron == 0:
                continue
            # Get block id for the layer
            for block, layer_list in blocks.items():
                if layer_name in layer_list:
                    block_idx = block_list.index(block)
                    break
                else:
                    block_idx = 0
                
            # Get values and channels remained
            # print(layer_name)
            layer_importance_value = metric[layer_name]
            # Important!!!!!!!!!!!!!!!! 8 hours of debugging on this.
            # print(mask)
            layer_importance_value = layer_importance_value[mask.bool()]
            # Align the channels for layers in the same group
            reaaranged_layer_importance_value = torch.gather(layer_importance_value, 0, group_sorted_indices)
            group_sorted_values = reaaranged_layer_importance_value.view(-1, group_size)
            combined_importance = group_sorted_values.sum(dim=1)
            # Set large values to avoid pruning
    #         if 'module.conv1' in group or ('module.features.conv_bn.conv' in group and int(value_remained.size(0)) <= 16):
    # #         if 'module.conv1' in group:
    #             combined_values[:] = 10000  # set to large values to avoid pruning
            for i in range(group_sorted_indice2.size(0)):
                # print(layer_name)
                used_layers.append(layer_name)
                grouped_neurons.append(
                    {
                        "layer_name": layer_name,
                        "channel_indices": [channels_remained[idx] for idx in group_sorted_indice2[i]],
                        "combined_importance": combined_importance[i].item(),
                        "block_number": block_idx,
                        "group_name": group,
                        "group_number": grouped_neuron_idx + i,
                    }
                )
        grouped_neuron_idx += group_sorted_indice2.size(0)

    # print(sorted(list(set(used_layers))))
    # exit()
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
    global used_keys
    if cin <= 0 or cout <= 0:
        return 0
    token_len = len(list(lookup_table.keys())[0].split('_'))
    if token_len == 6:
        key = '_'.join(
            [str(batch_size), str(int(cin)), str(int(cout)), str(fmap), str(k), str(stride)])
        # petr_key = '_'.join(
        #     [str(batch_size), str(int(cin)), str(int(cout)), str(fmap_mapping[fmap]), str(k), str(stride)])
        if key not in lookup_table:
            used_keys.append(key)
            near_p_count = 0
            summed_v = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    nkey = '_'.join([str(batch_size), str(int(cin+i)), str(int(cout+j)), str(fmap), str(k), str(stride)])
                    # nkey_petr = '_'.join([str(batch_size), str(int(cin+i)), str(int(cout+j)), str(fmap_mapping[fmap]), str(k), str(stride)])
                    if nkey in lookup_table:
                        near_p_count += 1
                        used_keys.append(nkey)
                        summed_v += lookup_table[nkey]
            if near_p_count == 0:
                print(key)
            lat_speed = summed_v / near_p_count
            # lat_speed = 1
        else:
            used_keys.append(key)
            lat_speed = lookup_table[key]
    else:
        key = '_'.join(
            [str(batch_size), str(int(cin)), str(int(cout)), str(fmap), str(k), str(stride), str(int(group_count))])
        # petr_key = '_'.join(
        #     [str(batch_size), str(int(cin)), str(int(cout)), str(fmap_mapping[fmap]), str(k), str(stride), str(int(group_count))])
        used_keys.append(key)
        lat_speed = lookup_table[key]
    # return 1
    return lat_speed


# Done Change
def get_total_latency(layers, groups, pre_group, layer_masks, fmap_table, lookup_table, lut_bs):
    """
        Get the approximated total latency for the conv network
    Args:
        layers (dict): {layer_name: the layer instance}
        groups (list): each item is a group name, which is a tuple containing all the layer names 
        pre_group (dict): {layer_name: the name of the connected former layer group}
        layer_masks (dict): {group_name: the channel mask}
        fmap_table (dict): the key is the layer name, the value is the feature map size
        lookup_table (dict): the key is <batch_size>_<cin>_<cout>_<fmap_size>_<k>_<stride>,
            the value is the corresponding latency
    Returns:
        total_latency (float): the approximated total latency
    """
    total_latency = 0
    for group in groups:
        # if layer_masks is not None and group in layer_masks:
        #     mask = layer_masks[group]
        #     active_neuron_num = torch.sum(mask.data).cpu().item()
        # else:
        #     active_neuron_num = layers[group[0]].weight.size(0)
        
        for layer_name in group:
            active_neuron_num = get_remaining_neuron_in_layer(layer_name, layers, layer_masks)
            if active_neuron_num == 0:
                continue
            pre_group_name = pre_group[layer_name]
            # the commented is wrong
            # if layer_masks is not None and pre_group_name in layer_masks:
            if layer_masks is not None and pre_group_name in groups:
                pre_active_neuron_num = get_remaining_neuron_in_group(pre_group_name, layers, layer_masks) if pre_group_name is not None else 3
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


def get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, layer_masks, fmap_table, lookup_table, lut_bs):
    group_latency_change = {}
    for group in groups:
        # latency change caused by the neuron num change in the pruned layers
        for layer_name in group:
            if layer_masks is not None and layer_name in layer_masks:
                mask = layer_masks[layer_name]
                active_neuron_num = int(torch.sum(mask.data).cpu().item())
            else:
                active_neuron_num = int(layers[layer_name].weight.size(0))
            channel_group_size = max([channel_group_size_dict[ln] for ln in group])
            channel_group_count = active_neuron_num//channel_group_size

            pre_group_name = pre_group[layer_name]
            if pre_group_name is None:
                pre_active_neuron_num = 3
            # elif layer_masks is not None and pre_group_name in layer_masks:
            elif layer_masks is not None and pre_group_name in groups:
                pre_active_neuron_num = get_remaining_neuron_in_group(pre_group_name, layers, layer_masks)
            else:
                pre_active_neuron_num = get_layer_neuron_num(layers[pre_group_name[0]])
                

            layer = layers[layer_name]
            k = layer.kernel_size[0]
            fmap = fmap_table[layer_name]
            stride = layer.stride[0]
            conv_groups = pre_active_neuron_num if layer.groups > 1 else 1

            if layer_name == "module.conv1":
                latency_change = np.zeros((channel_group_count + 1))
                for i in range(channel_group_count + 1):
                    if conv_groups == 1:
                        latency = get_layer_latency(lookup_table, pre_active_neuron_num, channel_group_size*i, k, fmap, stride, conv_groups, batch_size=lut_bs)
                    else:
                        print("Convolution Groups bigger than 1")
                        latency = get_layer_latency(lookup_table, channel_group_size*i, channel_group_size*i, k, fmap, stride, conv_groups-channel_group_size*i, batch_size=lut_bs)
                    # latency_change[i, j] = latency
                    latency_change[i] = round(latency*1000)
            else:
                pre_channel_group_size = max([channel_group_size_dict[ln] for ln in pre_group_name])
                pre_channel_group_count = pre_active_neuron_num//pre_channel_group_size
                pre_channel_group_count = int(pre_channel_group_count)
                # print(channel_group_count, pre_channel_group_count)
                latency_change = np.zeros((channel_group_count + 1, pre_channel_group_count + 1))
                for i in range(channel_group_count + 1):
                    for j in range(pre_channel_group_count + 1):
                        if conv_groups == 1:
                            latency = get_layer_latency(lookup_table, pre_channel_group_size*j, channel_group_size*i, k, fmap, stride, conv_groups, batch_size=lut_bs)
                        else:
                            print("Convolution Groups bigger than 1")
                            latency = get_layer_latency(lookup_table, channel_group_size*i, channel_group_size*i, k, fmap, stride, channel_group_size*i, batch_size=lut_bs)
                        # latency_change[i, j] = latency
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


def knapsack_pulp(weight, value, capacity, extra_space, channel_block_number, channel_group_number, layer_index_split, results_dir, blocks_num=None):
    assert len(weight) == len(value)
    print('++++++')
    print(capacity, sum(weight))
    if capacity >= sum(weight):
        return list(range(len(weight))), sum(weight), sum(value)

    ori_capacity = capacity
    capacity += extra_space

    # weight, value, layer_index_split = weight[::-1], value[::-1], [0]+list(itertools.accumulate(layer_index_split[::-1]))

    print(value)
    print(layer_index_split)
    n_items = len(value)

    num_channels = len(weight)
    num_blocks = max(channel_block_number) + 1
    num_grouped_channels = max(channel_group_number) + 1

    group_number = []
    for index, count in enumerate(layer_index_split):
        group_number += [index] * count

    layer_index_split = list(itertools.accumulate(layer_index_split))
    max_value_idx_per_group = [idx-1 for idx in layer_index_split]

    min_possible_capacity = 0
    for i in max_value_idx_per_group: 
        if channel_block_number[i] == 0:
            min_possible_capacity += weight[i]
            
    if capacity <= min_possible_capacity:
        print("Cannot prune further: Channel pruning limit reached. Target latency is increased to %f which is the minimum possible latency for this model with channel pruning.", min_possible_capacity)
        return max_value_idx_per_group
    print("Capacity greater than min_possible_capacity of %f", min_possible_capacity)


    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    # for layer_idx in range(len(layer_index_split)):
    #     pre_total_num = layer_index_split[layer_idx - 1] if layer_idx > 0 else 0
    #     layer_value = value[pre_total_num : layer_index_split[layer_idx]]
    #     sorted_value = sorted(layer_value)
    #     if layer_value != sorted_value:
    #         raise ValueError(
    #             "The importance of neuron groups from the same layer should be sorted ascendingly."
    #         )
    
    problem = LpProblem(name="resource-allocation", sense=LpMaximize)
    # Define the decision variables
    # Each variable is a binary value that 1 means being selected to keep while 0 means to prune.
    # decision_var = [LpVariable(name=f"y{i}", cat=LpBinary) for i in range(n_items)]

    decision_var = LpVariable.dicts("block_channels", [(i, b) for i in range(num_grouped_channels) for b in range(num_blocks)], cat='Binary')
    aux_var = LpVariable.dicts("auxilary", [(i, b) for i in range(len(layer_index_split)) for b in range(num_blocks)], cat='Binary')


    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. Preceding constraint: for neurons in the same layer, the more important one always
    # needs to be kept before a less important one'
    # 3. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    problem += (
        lpSum(weight[idx] * decision_var[(channel_group_number[idx], channel_block_number[idx])] for idx in range(num_channels)) <= capacity,
        "latency_constraint",
    )
    split_idx = 0
    # print(len(decision_var))
    # print(n_items)
    # print(layer_index_split)
    for idx in range(1, num_channels + 1):
        if idx != layer_index_split[split_idx]:
            problem += (
                decision_var[(channel_group_number[idx - 1], channel_block_number[idx - 1])] <= decision_var[(channel_group_number[idx], channel_block_number[idx])],
                f"preceding_constraint{idx}",
            )
        else:
            if channel_block_number[idx - 1] == 0:
                problem += (
                    decision_var[(channel_group_number[idx - 1], channel_block_number[idx - 1])] >= 1,
                    f"no_layer_prune_constraint{split_idx}",
                )
            split_idx += 1

    count = 0
    #Enforce Group Constraint
    # Group1: [Layer1, Layer3, Layer5, Layer 7]
    # Block0: Layer 1, Block1: Layer 3...
    for cnt, split_id in enumerate(layer_index_split):
        b_total = lpSum(decision_var[(channel_group_number[idx], 0)] for idx in range(count, split_id))
        for b_i in range(1, num_blocks):
            # Set up upperbounds for the number of channels in the prunable block
            for idx in range(count, split_id):
                problem += (decision_var[(channel_group_number[idx], b_i)] <= decision_var[(channel_group_number[idx], 0)])
            b_i_total = lpSum(decision_var[(channel_group_number[idx], b_i)] for idx in range(count, split_id))
            # b_total - b_i_total = 0 / b_total
            # split_id - count: total number of channels
            # <= 0, b_total = b_i_total
            problem += ((b_total - b_i_total) <= ((split_id - count) * aux_var[(cnt, b_i)]), f"equal_length_constraint_1_{cnt}_{b_i}")
            # b_i_total = 0
            problem += (b_i_total <= ((split_id - count) * (1-aux_var[(cnt, b_i)])), f"equal_length_constraint_2_{cnt}_{b_i}")
        count = split_id

    total_num_blocks = 0
    # Inside the same block, decisions should be unanimous
    for b in range(1, num_blocks):
        # channel indices for block b
        indices = [i for i, num in enumerate(channel_block_number) if num == b]
        # get a max value channel index to represent each layer
        last_group_idx_for_block = [last_group_idx for last_group_idx in max_value_idx_per_group if last_group_idx in indices]
        # for layers inside the same block, we should either keep them all or throw them all
        for tt in range(1, len(last_group_idx_for_block)):
            cur_id = channel_group_number[last_group_idx_for_block[tt]]
            prev_id = channel_group_number[last_group_idx_for_block[tt-1]]
            problem += decision_var[(prev_id, b)] == decision_var[(cur_id, b)]
            # if tt == 1:
            #     # We do not prune any of the block
            #     problem += decision_var[(prev_id, b)] == 1
            # Added a bounded number constraint
            if blocks_num is not None and blocks_num[0] and tt == 1:
                total_num_blocks += decision_var[(prev_id, b)]

    if blocks_num is not None and blocks_num[0]:
        problem += total_num_blocks <= blocks_num[1]


    # Set objective
    problem += lpSum(value[idx] * decision_var[(channel_group_number[idx], channel_block_number[idx])] for idx in range(num_channels))

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
    for i in range(num_channels):
        if decision_var[(channel_group_number[i], channel_block_number[i])].value() == 1:
            items_in_bag.append(i)

    items_binary = []
    for i in range(num_channels): 
        items_binary.append(decision_var[(channel_group_number[i], channel_block_number[i])].value())

    total_importance_after_pruning = 0
    for idx, importance in enumerate(value):
        total_importance_after_pruning += items_binary[idx] * importance

    importance_per_group_before_pruning = [0] * len(layer_index_split)
    importance_per_group_after_pruning = [0] * len(layer_index_split)
    neuron_groups_per_group_before_pruning = [0] * len(layer_index_split)
    neuron_groups_per_group_after_pruning = [0] * len(layer_index_split)

    for idx, group_num in enumerate(group_number):
        importance_per_group_before_pruning[group_num] += value[idx]
        importance_per_group_after_pruning[group_num] += items_binary[idx] * value[idx]
        neuron_groups_per_group_before_pruning[group_num] += 1
        neuron_groups_per_group_after_pruning[group_num] += items_binary[idx] * 1

    total_blocks = 0
    active_blocks = 0
    # Check Block Sparsity
    for b in range(1, num_blocks):
        # channel indices for block b
        indices = [i for i, num in enumerate(channel_block_number) if num == b]
        # get a max value channel index to represent each layer
        last_group_idx_for_block = [last_group_idx for last_group_idx in max_value_idx_per_group if last_group_idx in indices]
        # for layers inside the same block, we should either keep them all or throw them all
        total_blocks += 1
        try:
            cur_id = channel_group_number[last_group_idx_for_block[0]]
            active_blocks += decision_var[(cur_id, b)].value()
        except:
            continue

    print("Block Sparsity")
    print(f"Active / Total: {active_blocks} / {total_blocks}")
    # print("Finished Knapsack")
    # print("value")
    # print(value)
    # print("group number")
    # print(group_number)
    # print("items in bag")
    # print(items_binary)
    # print("importance per group before pruning")
    # print(importance_per_group_before_pruning)
    # print("importance per group after pruning")
    # print(importance_per_group_after_pruning)
    # print("neuron groups per layer group before pruning")
    # print(neuron_groups_per_group_before_pruning)
    # print("neuron groups per layer group after pruning")
    # print(neuron_groups_per_group_after_pruning)
    # print("total importance after pruning")
    # print(total_importance_after_pruning)

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
# ********************************************************************************************************************* #
def knapsack_pyomo(weight_dict, value_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, capacity, extra_space, layer_index_split, results_dir, init_values=None):
    print("Running Pyomo")
    print('++++++')

    ori_capacity = capacity
    capacity += extra_space

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    model = ConcreteModel()
    # Define Decision Variables
    all_block_items = list(sorted(list(set(layer2block.values()))))
#     all_block_items = ['b'+str(x) for x in all_block_items]
#     all_block_items = [1000+x for x in all_block_items]
#     model.block_decision_vars = Var(all_block_items, domain=Binary, initialize=1)
#     model.blocks = Var(all_block_items, domain=Binary, initialize=1)
#     print(len(model.block_decision_vars))
    group_var_slices = {}
    counter = 0
    for group_name, value in channel_group_count.items():
        # print(group_name)
        group_var_slices[group_name] = (counter, counter+value)
        counter += value
    counter_with_blocks = counter + len(all_block_items)
    # print(counter, counter_with_blocks)
    # print(group_var_slices)
    block_var_slices = {i:i+counter for i in range(len(all_block_items))}
    # print(block_var_slices)
#     all_items = list(range(counter))
    all_items = list(range(counter_with_blocks))

    # model.decision_vars = Var(all_items, domain=Binary)
    if init_values is not None:    
        model.decision_vars = Var(all_items, domain=Binary, initialize=init_values)
    else:
        model.decision_vars = Var(all_items, domain=Binary)
    # print(len(model.decision_vars))
    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    # 3. Only one unique configuration for one group
    # 4. Block 0 non prunable constraint
#     model.first_block_no_prune = Constraint(expr=model.block_decision_vars[0] == 1)
#     model.first_block_no_prune = Constraint(expr=model.decision_vars[block_var_slices[0]] == 1)
    model.first_block_no_prune = Constraint(expr=model.decision_vars[block_var_slices[0]] >= 1)
    latency_expr = 0
    importance = 0
    model.no_layer_prune_constraint = ConstraintList()
    model.group_unique_constraint = ConstraintList()
    used_blocks = []
    for group_name in channel_group_count.keys():
#         if group_name == ('module.layer1.0.downsample.0', 'module.layer1.0.conv3', 'module.layer1.1.conv3', 'module.layer1.2.conv3'):
#             print("set")
        # Get the decision variables
        cur_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[group_name][0], group_var_slices[group_name][1])]
        # Unique Constraint. We can only select one configuration.
        model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
#         model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) <= 1)
#         model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) >= 1)
#         model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
#         model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
        model.no_layer_prune_constraint.add(cur_decision_vars[0] <= 0)
        for layer_name in group_name:
            # Check for pruned layer
            if layer_name not in value_dict:
                print(f"{layer_name} belongs to pruned blocks. Skip...")
                continue
            block_id = layer2block[layer_name]
            # Latency Expression
            if layer_name == "module.conv1":
                pre_group_name, lat_vec = weight_dict[layer_name]
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
#                 model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] == 1)
                model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] >= 1)
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
#             latency_expr += model.block_decision_vars[block_id] * cur_expr

            used_blocks.append(block_id)
            latency_expr += model.decision_vars[block_var_slices[block_id]] * cur_expr
#             latency_expr += cur_expr

            # Importance Expression
#             importance += model.block_decision_vars[block_id] * sum(value_dict[layer_name][i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))
            importance += model.decision_vars[block_var_slices[block_id]] * sum(value_dict[layer_name][i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))
#             importance += sum(value_dict[layer_name][i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))
    
    print(set(used_blocks))
    # Latency Constraint
    model.latency_constraint = Constraint(expr=latency_expr <= capacity)
    # Block Number Constraint; For testing purposes for now
#     model.block_constraint = Constraint(expr=sum(model.decision_vars[block_var_slices[i]] for i in range(len(block_var_slices))) <= 10)
    # Set objective
    model.obj = Objective(expr=importance, sense=maximize)

    # Solve!
    # Mixed-Integer Nonlinear
    # model.obj.display()
    # model.display()
    # model.pprint()
    start = time.time()
    # solver = SolverFactory('mindtpy')
    solver = SolverFactory('mindtpy', nlp_solver='ipopt', nlp_solver_args={"executable": '/content/ipopt'}, mip_solver='glpk')
    # solver = SolverFactory('ipopt', executable='/content/ipopt')

    # solver = SolverFactory('glpk')
    # Without FP, problem is not solvable
    # if tries % 2 == 0:
    #     results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt')
    # else:
    #     results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt')

#     results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt', tee=True) 
    results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt', tee=False) 
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
    # Check how many number of channels to keep for each layer 
    for group_name in channel_group_count.keys():
        indices = list(range(group_var_slices[group_name][0], group_var_slices[group_name][1]))
        cur_decision_vars = [model.decision_vars[k] for k in indices]
        for i in range(len(cur_decision_vars)):
            # print(cur_decision_vars[i].value)
            # How many groups to keep?
            if cur_decision_vars[i].value == 1:
                if i == 0:
                    print(f"Unexpected, please Check! No channels for group {group_name}")
                num_channel_to_keep[group_name] = i

    print("num_channel_to_keep")
    print(num_channel_to_keep)
    # Get the indices for those kept channel groups
    cur_layer_name = None
    cur_start = None
    pruned_blocks = []
    for i, layer_name in enumerate(channel_layer_name):
        if i == 0:
            cur_layer_name = layer_name
            cur_start = 0
        if layer_name != cur_layer_name:
            cur_end = i
            # They should come from the same group
            cur_groups = channel_group_name[cur_start:cur_end]
            assert len(set(cur_groups)) == 1
            cur_group = cur_groups[0]
            num_to_keep = num_channel_to_keep[cur_group]
            # Override by pruned block
            cur_block_id = layer2block[cur_layer_name]
#             if model.block_decision_vars[cur_block_id].value == 0:
            if model.decision_vars[block_var_slices[cur_block_id]].value == 0:
                pruned_blocks.append(cur_block_id)
            else:
                keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
                items_in_bag += keep_idxs
            # Update layer in consideration
            cur_layer_name = layer_name
            cur_start = i 
    cur_end = len(channel_layer_name)
    cur_groups = channel_group_name[cur_start:cur_end]
#         print(cur_groups)
#             print(cur_groups)
    assert len(set(cur_groups)) == 1
    cur_group = cur_groups[0]
    num_to_keep = num_channel_to_keep[cur_group]
    cur_block_id = layer2block[cur_layer_name]
#     if model.block_decision_vars[cur_block_id].value == 0:
    if model.decision_vars[block_var_slices[cur_block_id]].value == 0:
        pruned_blocks.append(cur_block_id)
    else:
        keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
        items_in_bag += keep_idxs
    # Override by pruned block
    # keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
    # items_in_bag += keep_idxs
    
    print(f"Pruned {len(set(pruned_blocks))}/16 blocks in this step.")
    print(f"Pruned block ID: {set(pruned_blocks)}")
    print("Solve Time", end-start)
    print("Finished")
    # exit()
    # for group_name, value in value_dict.items():
    #     print(group_name)
    #     print(model.decision_vars[group_var_slices[group_name][0]:group_var_slices[group_name][1]])
    # print(items_in_bag)
    return items_in_bag, None, None

# ********************************************************************************************************************* #

def knapsack_pyomo_no_pruned_blocks(weight_dict, value_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, capacity, extra_space, layer_index_split, results_dir, init_values=None):
    print("Running Pyomo")
    print('++++++')

    ori_capacity = capacity
    capacity += extra_space

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    model = ConcreteModel()
    # Define Decision Variables
    group_var_slices = {}
    counter = 0
    for group_name, value in channel_group_count.items():
        # print(group_name)
        group_var_slices[group_name] = (counter, counter+value)
        counter += value
    
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
    for group_name in channel_group_count.keys():
        # Get the decision variables
        cur_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[group_name][0], group_var_slices[group_name][1])]
        # Unique Constraint. We can only select one configuration.
        model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
        for layer_name in group_name:
            # Latency Expression
            if layer_name == "module.conv1":
                pre_group_name, lat_vec = weight_dict[layer_name]
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] == 1)
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
            
            # Importance Expression
            importance += sum(value_dict[layer_name][i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))

    # Latency Constraint
    model.latency_constraint = Constraint(expr=latency_expr <= capacity)
    
    # Set objective
    model.obj = Objective(expr=importance, sense=maximize)

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
    # Check how many number of channels to keep for each layer 
    for group_name in channel_group_count.keys():
        indices = list(range(group_var_slices[group_name][0], group_var_slices[group_name][1]))
        cur_decision_vars = [model.decision_vars[k] for k in indices]
        for i in range(len(cur_decision_vars)):
            # print(cur_decision_vars[i].value)
            # How many groups to keep?
            if cur_decision_vars[i].value == 1:
                if i == 0:
                    print(f"No channels for group {group_name}")
                    break
                num_channel_to_keep[group_name] = i
    print("num_channel_to_keep")
    # print(num_channel_to_keep)
    # Get the indices for those kept channel groups
    cur_layer_name = None
    cur_start = None
    for i, layer_name in enumerate(channel_layer_name):
        if i == 0:
            cur_layer_name = layer_name
            cur_start = 0
        if layer_name != cur_layer_name:
            cur_end = i
            # They should come from the same group
            cur_groups = channel_group_name[cur_start:cur_end]
            assert len(set(cur_groups)) == 1
            cur_group = cur_groups[0]
            num_to_keep = num_channel_to_keep[cur_group]
            # Override by pruned block
            keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
            items_in_bag += keep_idxs
            # Update layer in consideration
            cur_layer_name = layer_name
            cur_start = i 
    cur_end = len(channel_layer_name)
    cur_groups = channel_group_name[cur_start:cur_end]
#         print(cur_groups)
#             print(cur_groups)
    assert len(set(cur_groups)) == 1
    cur_group = cur_groups[0]
    num_to_keep = num_channel_to_keep[cur_group]
    # Override by pruned block
    keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
    items_in_bag += keep_idxs
        
    print("Solve Time", end-start)
    print("Finished")
    # exit()
    # for group_name, value in value_dict.items():
    #     print(group_name)
    #     print(model.decision_vars[group_var_slices[group_name][0]:group_var_slices[group_name][1]])
    # print(items_in_bag)
    return items_in_bag, None, None

# ********************************************************************************************************************* #
def knapsack_pyomo_no_pruned_blocks_group(weight_dict, value_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, capacity, extra_space, layer_index_split, results_dir, init_values=None):
    print("Running Pyomo")
    print('++++++')

    ori_capacity = capacity
    capacity += extra_space

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    model = ConcreteModel()
    # Define Decision Variables
    group_var_slices = {}
    counter = 0
    for group_name, value in channel_group_count.items():
        # print(group_name)
        group_var_slices[group_name] = (counter, counter+value)
        counter += value
    
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
    for group_name in channel_group_count.keys():
        # Get the decision variables
        cur_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[group_name][0], group_var_slices[group_name][1])]
        # Unique Constraint. We can only select one configuration.
        model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
        importance_Val = None
        for layer_name in group_name:
            if importance_Val is None:
                importance_Val = value_dict[layer_name]
            else:
                importance_Val = [importance_Val[i] + value_dict[layer_name][i] for i in range(len(importance_Val))]

            # Latency Expression
            if layer_name == "module.conv1":
                pre_group_name, lat_vec = weight_dict[layer_name]
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] == 1)
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
        print(group_name, importance_Val)
        # Importance Expression
        importance += sum(importance_Val[i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))

    # Latency Constraint
    model.latency_constraint = Constraint(expr=latency_expr <= capacity)
    
    # Set objective
    model.obj = Objective(expr=importance, sense=maximize)

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
    # Check how many number of channels to keep for each layer 
    for group_name in channel_group_count.keys():
        indices = list(range(group_var_slices[group_name][0], group_var_slices[group_name][1]))
        cur_decision_vars = [model.decision_vars[k] for k in indices]
        for i in range(len(cur_decision_vars)):
            # print(cur_decision_vars[i].value)
            # How many groups to keep?
            if cur_decision_vars[i].value == 1:
                if i == 0:
                    print(f"No channels for group {group_name}")
                    break
                num_channel_to_keep[group_name] = i
    print("num_channel_to_keep")
    # print(num_channel_to_keep)
    # Get the indices for those kept channel groups
    cur_layer_name = None
    cur_start = None
    for i, layer_name in enumerate(channel_layer_name):
        if i == 0:
            cur_layer_name = layer_name
            cur_start = 0
        if layer_name != cur_layer_name:
            cur_end = i
            # They should come from the same group
            cur_groups = channel_group_name[cur_start:cur_end]
            assert len(set(cur_groups)) == 1
            cur_group = cur_groups[0]
            num_to_keep = num_channel_to_keep[cur_group]
            # Override by pruned block
            keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
            items_in_bag += keep_idxs
            # Update layer in consideration
            cur_layer_name = layer_name
            cur_start = i 
    cur_end = len(channel_layer_name)
    cur_groups = channel_group_name[cur_start:cur_end]
#         print(cur_groups)
#             print(cur_groups)
    assert len(set(cur_groups)) == 1
    cur_group = cur_groups[0]
    num_to_keep = num_channel_to_keep[cur_group]
    # Override by pruned block
    keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
    items_in_bag += keep_idxs
        
    print("Solve Time", end-start)
    print("Finished")
    # exit()
    # for group_name, value in value_dict.items():
    #     print(group_name)
    #     print(model.decision_vars[group_var_slices[group_name][0]:group_var_slices[group_name][1]])
    # print(items_in_bag)
    return items_in_bag, None, None
# ********************************************************************************************************************* #

def apply_latency_target_pruning(output_dir, layers, layer_bn, blocks, layer_gate, metric, target_latency, layer_masks,
                                 groups, pre_group, aft_group_list, fmap_table, lookup_table, method, 
                                 wrap_solver=None, mu=0., step_size=-1, lut_bs=256, pulp=False, blocks_num=None, no_blockprune=False):
    """
        Apply latency targeted pruning to ensure the pruned network under a certain latency constraint.
    Args:
        layers (dict): {layer_name: the layer instance}
        layer_bn (dict): {layer_name: the corrsponding following bn layer name}
        layer_gate (dict): {layer_name: the corresponding following gate layer name}
        metric (dict): the metric to measure the importance of the neurons, {group_name: tensor of neuron importance}
        target_latency (float): the targeted latency
        layer_masks (dict): {group_name: the channel mask}
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

    # Done Change
    # group_neurons_inputs = [metric, layer_masks, layers, groups, blocks]
    # with open(f"{prune_counter}_group_inputs_hahp_latv2.pkl", 'wb') as f:
    #     pickle.dump(group_neurons_inputs, f)
    grouped_neurons = group_neurons_by_rank(metric, layer_masks, layers, groups, blocks)
    # {group_name: [latency_change]}
    # in each layer, the list of latency change is calculated by decreasing the out channel gradually, adaptively

    # Done Change
    adaptive_group_latency_change = get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, layer_masks, fmap_table, lookup_table, lut_bs)
    latency_dict = adaptive_group_latency_change
    print("Got the adaptive_group_latency_change")
    # print(adaptive_group_latency_change)
    # grouped_neuron_idx_list = range(len(list(grouped_neurons.keys())))
    grouped_neuron_idx_list = range(len(grouped_neurons))
    importance_dict = {}
    layer2group = {}
    # {
    #     "layer_name": ,
    #     "channel_indices": ,
    #     "combined_importance": ,
    #     "block_number": ,
    #     "group_name": ,
    #     "group_number": ,
    # }
    # Group Neurons are in ascending order
    for idx in grouped_neuron_idx_list:
        group_name = grouped_neurons[idx]["group_name"]
        layer_name = grouped_neurons[idx]["layer_name"]
        # print(layer_name, grouped_neurons[idx]["combined_importance"])
        layer2group[layer_name] = group_name
        cur_list = importance_dict.get(layer_name, [])
        # cur_list.append(round(grouped_neurons[idx]["combined_importance"]*1000000))
        cur_list.append(round(grouped_neurons[idx]["combined_importance"]*1000000))
        importance_dict[layer_name] = cur_list
    
    print("Before group combinging.")
    # for key, val in importance_dict.items():
    #     print(key, val)

    print("After group combining.")

    for key in importance_dict:
        importance_dict[key] = list(itertools.accumulate(importance_dict[key][::-1]))
        # None selected for this layer
        importance_dict[key] = [0] + importance_dict[key]
    
    combined_importance_dict = {}
    for key, val in importance_dict.items():
        group = layer2group[key]
        if group not in combined_importance_dict:
            combined_importance_dict[group] = val
        else:
            cur_val = combined_importance_dict[group]
            combined_importance_dict[group] = [cur_val[i]+val[i] for i in range(len(val))]
    
    for key, val in combined_importance_dict.items():
        print(key, val)
    # for key, val in importance_dict.items():
    #     print(key, val)

    # importance_list = [round(grouped_neurons[idx]["combined_importance"]*1000000) for idx in grouped_neuron_idx_list]
    channel_block_list = [grouped_neurons[idx]["block_number"] for idx in grouped_neuron_idx_list]
    channel_group_number = [grouped_neurons[idx]["group_number"] for idx in grouped_neuron_idx_list]
    channel_group_name = [grouped_neurons[idx]["group_name"] for idx in grouped_neuron_idx_list]
    channel_layer_name = [grouped_neurons[idx]["layer_name"] for idx in grouped_neuron_idx_list]
    channel_group_count = {}
    for i in range(len(channel_group_name)):
        group_name = channel_group_name[i]
        group_number = channel_group_number[i]
        if group_name not in channel_group_count:
            channel_group_count[group_name] = []
        channel_group_count[group_name].append(group_number)
    for group_name in channel_group_count.keys():
        # +1 because we also want to represent zero group selected
        channel_group_count[group_name] = max(channel_group_count[group_name]) - min(channel_group_count[group_name]) + 1 + 1

    # print(channel_group_number)
    # print(channel_group_name)
    # print(channel_group_count)
    layer2block = {}
    block_list = sorted(blocks.keys())
    for group in groups:
        for layer_name in group:
            for block, layer_list in blocks.items():
                if layer_name in layer_list:
                    block_idx = block_list.index(block)
                    break
                else:
                    block_idx = 0
            layer2block[layer_name] = block_idx
    # print(layer2block)

    # print(channel_layer_name)
    # print(grouped_neuron_idx_list)
    # print(len(grouped_neuron_idx_list))
    # print(adaptive_group_latency_change.keys())
    # print(len(adaptive_group_latency_change.keys()))
    # latency_list = [int(item*100) for idx in grouped_neuron_idx_list for item in adaptive_group_latency_change[grouped_neurons[idx]["layer_group_name"]]]
    
    print("Adaptive group latency change")
    # print(adaptive_group_latency_change)
    # print(sorted(list(adaptive_group_latency_change.keys())))

    # print("Latency List")
    # print(latency_list)
    # print("Latency Summation")
    # print(sum(latency_list))
    total_latency = get_total_latency(layers, groups, pre_group, layer_masks, fmap_table, lookup_table, lut_bs)
    prev_latency = total_latency
    print(f"total_latency:{total_latency}")
    extra_space = 0
    layer_index_split = None
    inputs = [latency_dict, importance_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, round(target_latency*1000), extra_space, layer_index_split, output_dir]
    with open(f"{prune_counter}_solver_inputs_hahp_latv2_testing.pkl", 'wb') as f:
        pickle.dump(inputs, f)

    # for key, val in importance_dict.items():
    #     print("Scaling down for easier handling")
    #     importance_dict[key] = [x//10 for x in val]
    # global used_keys
    # all_used_keys = list(set(used_keys))
    # with open(f"used_lut_keys_32.pkl", 'wb') as f:
    #     pickle.dump(all_used_keys, f)
    # exit()

    tries = 0
    # items_in_bag, _, _ = knapsack_pyomo(latency_dict, importance_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=None)
    items_in_bag, _, _ = knapsack_pyomo_no_pruned_blocks(latency_dict, importance_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=None)
    # while tries < 6:
    #     try:
    #         if tries == 1:
    #             init_values = 0
    #             with open("failed_solver_inputs.pkl", 'wb') as f:
    #                 pickle.dump(inputs, f)
    #             exit()
    #         elif tries == 2:
    #             init_values = 1
    #         else:
    #             init_values = None
            
    #         if tries >= 3:
    #             for key, val in importance_dict.items():
    #                 print("Scaling down for easier handling")
    #                 importance_dict[key] = [x//1000 for x in val]
    #         if no_blockprune:
    #             items_in_bag, _, _ = knapsack_pyomo_no_pruned_blocks(latency_dict, importance_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=init_values)
    #         else:
    #             print("Scaling down for easier handling")
    #             # for key, val in importance_dict.items():
    #             #     importance_dict[key] = [round(x / 1000.0) for x in val]
    #             items_in_bag, _, _ = knapsack_pyomo(latency_dict, importance_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, round(target_latency*1000), extra_space, layer_index_split, output_dir, init_values=init_values)
    #         break
    #     except:
    #         print("error. trying again...")
    #         tries += 1
    #         if tries == 5:
    #         # if tries == 1:
    #             print("Not pruning this step")
    #             items_in_bag = grouped_neuron_idx_list
    #             break

    # get the pruned dict
    pruned_items = set(grouped_neuron_idx_list) - set(items_in_bag)
    pruned_dict = {}
    for pruned_idx in list(pruned_items):
        layer_name = grouped_neurons[pruned_idx]["layer_name"]
        pruned_channel_sum = len(grouped_neurons[pruned_idx]["channel_indices"])
        if layer_name not in pruned_dict:
            pruned_dict[layer_name] = 0
        pruned_dict[layer_name] += pruned_channel_sum
    print("Pruned Distribution", pruned_dict)
    # with open(f"hahp_latv2_pruned_dict{prune_counter}_testing.pkl", 'wb') as f:
    #     pickle.dump(pruned_dict, f)
    # prune_counter += 1

    # reset the mask to 0
    ori_channel_num = {}
    for layer_name in metric.keys():
        ori_channel_num[layer_name] = int(torch.sum(layer_masks[layer_name]).item())
        layer_masks[layer_name][:] = 0.

    # set the selected neurons to active 
    for item_idx in items_in_bag:
        layer_name = grouped_neurons[item_idx]["layer_name"]
        # print("Kept", layer_name)
        channel_indices = grouped_neurons[item_idx]["channel_indices"]
        layer_masks[layer_name][channel_indices] = 1.

    total_pruned_num = 0
    for layer_name, mask in layer_masks.items():
        cur_channel_num = int(torch.sum(mask).item())
        pruned_channel = ori_channel_num[layer_name] - cur_channel_num
        total_pruned_num += pruned_channel*len(layer_name)
        # print('*** Group {}: {} channels / {} neurons are pruned at current step. '
        #       '{} channels / {} neurons left. ***'.format(layer_name,
        #                                                   pruned_channel,
        #                                                   pruned_channel*len(group_name),
        #                                                   cur_channel_num,
        #                                                   cur_channel_num*len(group_name)))

    # mask conv and bn parameters 
    # mask_conv_bn(layers, layer_bn, layer_gate, layer_masks)

    # get the total latency after pruning
    total_latency = get_total_latency(layers, groups, pre_group, layer_masks, fmap_table, lookup_table, lut_bs)
    print(f"Prev Latency after pruning: {prev_latency}")
    print(f"Actual Latency after pruning: {total_latency}")

    # with open(f"{prune_counter}_pruned_mask.pkl", 'wb') as f:
    #     pickle.dump(layer_masks, f)
    prune_counter += 1
    # print('Achieved latency: {}, actual latency after pruning: {}'.format(used_capacity, total_latency))
    # exit()

    return total_pruned_num
    
def mask_conv_bn(layers, layer_bn, layer_gate, layer_masks):
    """
        Mask the pruned neurons, set the corresponding neurons to zero
    Args:
        layers (dict): {layer_name: the layer instance}
        layer_bn (dict): {layer_name: the corrsponding following bn layer name}
        layer_gate (dict): {layer_name: the corresponding following gate layer name}
        layer_masks (dict): {group_name: the channel mask}
    """
    for layer_name, mask in layer_masks.items():
        if layer_gate is not None:
            gate_layer = layers[layer_gate[layer_name]]
            gate_layer.weight.data.mul_(mask)

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
