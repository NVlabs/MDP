import torch
import math
import itertools
import numpy as np
from collections import defaultdict
from utils.utils import count_non_zero_neurons
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
from prune.mdp_func import *
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

def load_group_size(arch):
    global channel_group_size_dict
    with open('group_size/{}_group_size.json'.format(arch), 'r') as f:
        channel_group_size_dict = json.load(f)

# def load_group_size(arch):
#     global channel_group_size_dict
#     with open('group_size/{}_group_size_32.json'.format(arch), 'r') as f:
#         channel_group_size_dict = json.load(f)

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
        importance_value = 0
        for layer_name in group:
            importance_value += metric[layer_name]
        # perform mean on importance value [Optionally]

        _, mask = get_mask_for_group(group, layers, layer_masks)
        
        if mask is not None:
            value_remained = importance_value[mask == 1.0]
            channels_remained = np.arange(mask.size(0))[mask.cpu().numpy() == 1.0]
        else:
            for layer_name in group:
                layer_masks[layer_name] = torch.ones_like(importance_value)
            value_remained = importance_value
            channels_remained = np.arange(layer_masks[group[0]].size(0))
        
        assert 'module.conv1' in group or int(value_remained.size(0)) % max([channel_group_size_dict[ln] for ln in group]) == 0
        sorted_values, sorted_indices = torch.sort(value_remained)
        # sorted_values = sorted_values.view(-1, CHANNEL_GROUP_SIZE)
        # sorted_indices = sorted_indices.view(-1, CHANNEL_GROUP_SIZE)
        group_size = max([channel_group_size_dict[ln] for ln in group])
        group_sorted_values = sorted_values.view(-1, group_size)
        group_sorted_indices = sorted_indices.view(-1, group_size)

        # print("group", group)
        for layer_name in group:
            active_neuron, layer_mask = get_mask_for_layer(layer_name, layers, layer_masks)
            if active_neuron == 0:
                continue
            for block, layer_list in blocks.items():
                if layer_name in layer_list:
                    block_idx = block_list.index(block)
                    break
                else:
                    block_idx = 0
            
            # print(layer_name)
            layer_importance_value = metric[layer_name]
            # Important!!!!!!!!!!!!!!!! 8 hours of debugging on this.
            layer_importance_value = layer_importance_value[mask.bool()]
            reaaranged_layer_importance_value = torch.gather(layer_importance_value, 0, sorted_indices)
            group_sorted_values = reaaranged_layer_importance_value.view(-1, group_size)
            combined_values = group_sorted_values.sum(dim=1)
            if 'module.conv1' in group or ('module.features.conv_bn.conv' in group and int(value_remained.size(0)) <= 16):
    #         if 'module.conv1' in group:
                combined_values[:] = 10000  # set to large values to avoid pruning
            for i in range(group_sorted_indices.size(0)):
                # grouped_neurons[grouped_neuron_idx + i] = \
                #     {
                #         "layer_name": layer_name,
                #         "channel_indices": [
                #             channels_remained[idx] for idx in group_sorted_indices[i]
                #         ],
                #         "combined_importance": combined_values[i].item(),
                #         "block_number": block_idx,
                #         "group_name": group,
                #         "group_number": grouped_neuron_idx + i,
                #     }
                used_layers.append(layer_name)
                grouped_neurons.append(
                    {
                        "layer_name": layer_name,
                        "channel_indices": [
                            channels_remained[idx] for idx in group_sorted_indices[i]
                        ],
                        "combined_importance": combined_values[i].item(),
                        "block_number": block_idx,
                        "group_name": group,
                        "group_number": grouped_neuron_idx + i,
                    }
                )
        grouped_neuron_idx += group_sorted_indices.size(0)

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


# Done Change
def get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, layer_masks, fmap_table, lookup_table, lut_bs):
    layer_latency_change = {}
    for group in groups:
        for layer_name in group:
            active_neuron_num = get_remaining_neuron_in_layer(layer_name, layers, layer_masks)

            channel_group_size = max([channel_group_size_dict[ln] for ln in group])
            channel_group_count = active_neuron_num//channel_group_size
            latency_change = [0 for _ in range(channel_group_count)]
            # the commented is wrong
            # latency change caused by the neuron num change in the pruned layers
            
            pre_group_name = pre_group[layer_name]
            if pre_group_name is None:
                pre_active_neuron_num = 3
            # elif layer_masks is not None and pre_group_name in layer_masks:
            elif layer_masks is not None and pre_group_name in groups:
                pre_active_neuron_num = get_remaining_neuron_in_group(pre_group_name, layers, layer_masks)
            else:
                pre_active_neuron_num = get_layer_neuron_num(
                        layers[pre_group_name[0]]
                    )
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
                layer_latency_diff = latency - reduced_latency
                latency_change[i] += layer_latency_diff
            #latency_change = [max(item, 0) for item in latency_change]

            layer_latency_change[layer_name] = latency_change

        # latency change is calculated by decreasing the output channel num,
        # thus the first calculated latency change of this layer should corresponds to the neuron(s) with least importance score.

    return layer_latency_change


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
            #     # For testing:
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


def apply_latency_target_pruning(output_dir, layers, layer_bn, blocks, layer_gate, metric, target_latency, layer_masks,
                                 groups, pre_group, aft_group_list, fmap_table, lookup_table, method, 
                                 wrap_solver=None, mu=0., step_size=-1, lut_bs=256, pulp=False, blocks_num=None):
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
    if step_size > 0:
        for k in channel_group_size_dict.keys():
            channel_group_size_dict[k] = min(step_size, int(layers[k].weight.size(0)))
    
    # adaptive latency change
    # {grouped_neuron_idx: {"layer_group_name": name, "channel_indices": [idx], "combined_importance": value}}
    # in each layer, the neuron group with higher importance score has larger grouped_neuron_idx

    # Done Change
    grouped_neurons = group_neurons_by_rank(metric, layer_masks, layers, groups, blocks)
    # {group_name: [latency_change]}
    # in each layer, the list of latency change is calculated by decreasing the out channel gradually, adaptively

    # Done Change
    adaptive_group_latency_change = get_adaptive_group_latency_contribute(layers, groups, pre_group, aft_group_list, layer_masks, fmap_table, lookup_table, lut_bs)
    # print(adaptive_group_latency_change)
    # grouped_neuron_idx_list = range(len(list(grouped_neurons.keys())))
    grouped_neuron_idx_list = range(len(grouped_neurons))
    importance_list = [round(grouped_neurons[idx]["combined_importance"]*1000000) for idx in grouped_neuron_idx_list]
    channel_block_list = [grouped_neurons[idx]["block_number"] for idx in grouped_neuron_idx_list]
    channel_group_number = [grouped_neurons[idx]["group_number"] for idx in grouped_neuron_idx_list]
    channel_layer_name = [grouped_neurons[idx]["layer_name"] for idx in grouped_neuron_idx_list]
    # print(channel_layer_name)
    # print(grouped_neuron_idx_list)
    # print(len(grouped_neuron_idx_list))
    # print(adaptive_group_latency_change.keys())
    # print(len(adaptive_group_latency_change.keys()))
    # latency_list = [int(item*100) for idx in grouped_neuron_idx_list for item in adaptive_group_latency_change[grouped_neurons[idx]["layer_group_name"]]]
    
    print("Adaptive group latency change")
    # print(adaptive_group_latency_change)
    # print(sorted(list(adaptive_group_latency_change.keys())))

    layer_index_split = []
    cur_gn, cnt = grouped_neurons[0]['layer_name'], 0
    latency_list = [round(item*1000) for item in adaptive_group_latency_change[cur_gn]]
    used_layer_names = [cur_gn]
    print(latency_list)
    for idx in grouped_neuron_idx_list:
        if grouped_neurons[idx]['layer_name'] == cur_gn:
            cnt += 1
        else:
            layer_index_split.append(cnt)
            cur_gn, cnt = grouped_neurons[idx]['layer_name'], 1
            used_layer_names.append(cur_gn)
            latency_list.extend([round(item*1000) for item in adaptive_group_latency_change[cur_gn]])
            print([round(item*1000) for item in adaptive_group_latency_change[cur_gn]])
    # print("Used layer names")
    # print(sorted(used_layer_names))
    layer_index_split.append(cnt)
    assert sum(layer_index_split) == len(grouped_neurons)
    # print("Latency List")
    # print(latency_list)
    # print("Latency Summation")
    # print(sum(latency_list))
    neg_latency = [item for item in latency_list if item < 0]
    extra_space = 0 if len(neg_latency)==0 else abs(sum(neg_latency))

    total_latency = get_total_latency(layers, groups, pre_group, layer_masks, fmap_table, lookup_table, lut_bs)
    prev_latency = total_latency
    summed_latency = sum([sum(adaptive_group_latency_change[group_name]) for group_name in adaptive_group_latency_change.keys()])
    print(f"total_latency:{total_latency}, summed_latency_contribute:{summed_latency}")
    print(f"neg space: {extra_space}")
    
    layer_importance_dump = {}
    layer_latency_dump = {}
    for idx in grouped_neuron_idx_list:
        layer_name = grouped_neurons[idx]["layer_name"]
        score = grouped_neurons[idx]["combined_importance"]
        if layer_name not in layer_importance_dump:
            layer_importance_dump[layer_name] = 0
        layer_importance_dump[layer_name] += score
    for layer_name in adaptive_group_latency_change.keys():
        if layer_name not in layer_latency_dump:
            layer_latency_dump[layer_name] = 0
        layer_latency_dump[layer_name] += sum(adaptive_group_latency_change[layer_name]) 
    print("Layer Importance Dump")
    print(layer_importance_dump)
    print("Layer Latency Dump")
    print(layer_latency_dump)
    global prune_counter
    with open(f"layer_importance_dump{prune_counter}.pkl", 'wb') as f:
        pickle.dump(layer_importance_dump, f)
    with open(f"layer_latency_dump{prune_counter}.pkl", 'wb') as f:
        pickle.dump(layer_latency_dump, f)
    if pulp:
        print("Running pulp solver for knapsack")
        items_in_bag, _, _ = knapsack_pulp(latency_list, importance_list, round(target_latency*1000), extra_space, channel_block_list, channel_group_number, layer_index_split, output_dir, blocks_num)
    else:
        items_in_bag, used_capacity, achieved_value = knapsack_dp_adaptive(latency_list, importance_list, round(target_latency*1000), extra_space, layer_index_split)

        # used_capacity /= 1000
    pruned_items = set(grouped_neuron_idx_list) - set(items_in_bag)
    pruned_dict = {}
    for pruned_idx in list(pruned_items):
        layer_name = grouped_neurons[pruned_idx]["layer_name"]
        pruned_channel_sum = len(grouped_neurons[pruned_idx]["channel_indices"])
        if layer_name not in pruned_dict:
            pruned_dict[layer_name] = 0
        pruned_dict[layer_name] += pruned_channel_sum
    print("Pruned Distribution", pruned_dict)
    with open(f"pruned_dict{prune_counter}.pkl", 'wb') as f:
        pickle.dump(pruned_dict, f)
    prune_counter += 1
    
    # reset the mask to 0
    ori_channel_num = {}
    for layer_name in metric.keys():
        ori_channel_num[layer_name] = int(torch.sum(layer_masks[layer_name]).item())
        layer_masks[layer_name][:] = 0.

    # set the selected neurons to active 
    for item_idx in items_in_bag:
        layer_name = grouped_neurons[item_idx]["layer_name"]
        channel_indices = grouped_neurons[item_idx]["channel_indices"]
        layer_masks[layer_name][channel_indices] = 1.

    total_pruned_num = 0
    for layer_name, mask in layer_masks.items():
        cur_channel_num = int(torch.sum(mask).item())
        pruned_channel = ori_channel_num[layer_name] - cur_channel_num
        total_pruned_num += pruned_channel*len(layer_name)
        # print('*** Group {}: {} channels / {} neurons are pruned at current step. '
        #       '{} channels / {} neurons left. ***'.format(group_name,
        #                                                   pruned_channel,
        #                                                   pruned_channel*len(group_name),
        #                                                   cur_channel_num,
        #                                                   cur_channel_num*len(group_name)))

    # mask conv and bn parameters 
    mask_conv_bn(layers, layer_bn, layer_gate, layer_masks)

    # get the total latency after pruning
    total_latency = get_total_latency(layers, groups, pre_group, layer_masks, fmap_table, lookup_table, lut_bs)
    print(f"Prev Latency after pruning: {prev_latency}")
    print(f"Actual Latency after pruning: {total_latency}")

    # print('Achieved latency: {}, actual latency after pruning: {}'.format(used_capacity, total_latency))

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
