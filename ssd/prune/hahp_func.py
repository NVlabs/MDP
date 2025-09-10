import torch
import torch.nn as nn
# def build_block_from_layers(layers):
#     layer_list = list(layers.keys())
#     print(layer_list)
#     # module.layer3.4.bn3
#     blocks = {}
#     keyword = 'blocks'
#     for layer in layer_list:
#         split_layer_name = layer.split('.')
#         if keyword in split_layer_name:
#             idx = split_layer_name.index(keyword)
#             block_name = '.'.join(split_layer_name[:idx+3])
#             blocks.setdefault(block_name, []).append(layer)
#         # else:
#         #    blocks.setdefault(layer,[]).append(layer)
#     return blocks

# 3, 4, 6, 3
def build_block_from_layers(layers):
    layer_list = list(layers.keys())
    print(layer_list)
    # module.layer3.4.bn3
    blocks = {}
    for layer in layer_list:
        if 'f_0' not in layer:
            continue
        layer2 = layer.replace('f_0.', '')
        split_layer_name = layer2.split('.')
        if len(split_layer_name) == 4:
            block_name = split_layer_name[1] + "." + split_layer_name[2]
            blocks.setdefault(block_name, []).append(layer)
        # else:
        #    blocks.setdefault(layer,[]).append(layer)

    # Nonprunable block
    # Comment the following line if you want to keep the actual first block in ResNet50 unprunable
    blocks["layer0.0"] = []
    return blocks


def get_remaining_neuron_in_layer(layer, layers, layer_masks):
    if layer in layer_masks:
        active_neuron_num = int(layer_masks[layer].data.sum().item())
    else:
        active_neuron_num = layers[layer].weight_orig.size(0)

    return active_neuron_num


def get_remaining_neuron_in_group(group, layers, layer_masks):
    active_neuron_num = get_remaining_neuron_in_layer(group[0], layers, layer_masks)
    for layer in group:
        active_neuron_num_tmp = get_remaining_neuron_in_layer(layer, layers, layer_masks)
        if active_neuron_num_tmp == 0:
            continue
        elif active_neuron_num_tmp != active_neuron_num:
            if active_neuron_num == 0:
                active_neuron_num = active_neuron_num_tmp
            else:
                raise ValueError(f"Group mask do not match for layers in {group}.")

    return active_neuron_num


def get_layer_neuron_num(layer: torch.nn.Module) -> int:
    """Get the neuron number in the given layer.

    Args:
        layer (torch.nn.Module): A PyTorch layer module.
    Returns:
        neuron_num (int): The number of neurons.
    """
    if isinstance(layer, nn.ConvTranspose2d):
        neuron_num = int(layer.weight_orig.size(1))
    else:
        neuron_num = int(layer.weight_orig.size(0))
    return neuron_num


def get_mask_for_layer(layer, layers, layer_masks):
    if layer in layer_masks:
        active_neuron_num = int(layer_masks[layer].data.sum().item())
        return active_neuron_num, layer_masks[layer]
    else:
        active_neuron_num = get_layer_neuron_num(layers[layer])
        return active_neuron_num, torch.ones(active_neuron_num)


def get_mask_for_group(group, layers, layer_masks):
    active_neuron_num, mask = get_mask_for_layer(group[0], layers, layer_masks)
    for layer in group:
        active_neuron_num_tmp, mask_tmp = get_mask_for_layer(layer, layers, layer_masks)
        if active_neuron_num_tmp == 0:
            continue
        elif active_neuron_num_tmp != active_neuron_num:
            if active_neuron_num == 0:
                active_neuron_num = active_neuron_num_tmp
                mask = mask_tmp
            else:
                raise ValueError(f"Group mask do not match for layers in {group}.")

    return active_neuron_num, mask