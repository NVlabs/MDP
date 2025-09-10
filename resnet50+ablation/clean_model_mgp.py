import os
import argparse
import pickle as pkl
import json

import torch
#from thop import profile, clever_format
from models.resnet_pruned import ResNet34, ResNet50, ResNet101
from models.mobilenet_pruned import mobilenet
from models.vgg_pruned import vgg16
from models.models import get_model
from prune.prune_config import PruneConfigReader
from prune.mdp_func import *
from utils.utils import args_parser, extract_layers, is_conv
from copy import deepcopy

def args_parser():
    parser = argparse.ArgumentParser(description='create the network and test the flop')
    parser.add_argument('-a', '--arch', default='resnet50', help='the network architecture')
    parser.add_argument('--dataset-name', type=str, default='ImageNet',
                        help='The name of the dataset')
    parser.add_argument('--enable-bias', type=bool, default=False,
                        help='Whether to enable the bias term in the convolution layers')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate layers in the network')

    parser.add_argument('--pre-trained', type=str, default=None,
                        help='the path of a pre-trained model if want to start from some checkpoint')
    parser.add_argument('--mask', type=str, default=None,
                        help='the mask file')
    parser.add_argument('--layer-size-file', type=str, default=None, help='file saving size of each layer')

    args = parser.parse_args()
    return args


def main(args):
    adjacency_file = './net_structure/{}_adjacency.json'.format(args.arch)
    assert os.path.exists(adjacency_file)
    with open(adjacency_file, 'r') as f:
        adjacency = json.load(f)

    from prune.prune_config import PruneConfigReader
    config_reader = PruneConfigReader()
    config_reader.set_prune_setting("configs/resnet50.json")
    layer_structure = config_reader.get_layer_structure()  # conv_bn, conv_gate, groups
    conv_bn, conv_gate, groups = layer_structure
    from prune.prune_config_with_structure import PruneConfigReader
    config_reader = PruneConfigReader()
    config_reader.set_prune_setting("configs/resnet50_structure.json")
    _, _, _, pre_group, aft_group_list = config_reader.get_layer_structure()

    resume_weights = {}
    if args.pre_trained is not None and os.path.isfile(args.pre_trained):
        if args.pre_trained.endswith('.pth'):
            print("=> loading checkpoint weights from '{}'".format(args.pre_trained))
            checkpoint = torch.load(args.pre_trained, map_location=lambda storage, loc: storage)
            resume_weights = checkpoint
        elif args.pre_trained.endswith('.tar'):
            print("=> loading checkpoint weights and optimizer from '{}'".format(args.pre_trained))
            checkpoint = torch.load(args.pre_trained, map_location=lambda storage, loc: storage)
            resume_weights = checkpoint['model_state_dict']
        else:
            print("=> unsupported file for loading at '{}'".format(args.pre_trained))
    else:
        print("=> no checkpoint weights found at '{}'".format(args.pre_trained))
    tmp = {}
    for k, v in resume_weights.items():
        if 'module.' not in k:
            tmp['module.'+k]=v
        else:
            tmp[k] = v
    resume_weights = tmp

    num_class = 1000 if args.dataset_name == 'ImageNet' else 10
    layer_num_dict = {}
    layer_to_group = {}
    total_num = 0
    layer_pruned = []
    layer_pruned2 = []
    block_pruned = []
    # module.layer3.0.conv3
    if args.mask is not None:
        with open(args.mask, 'rb') as f:
            layer_masks = pkl.load(f)
        for layer_name, mask in layer_masks.items():
            group_neuron_remain = int(torch.sum(mask).item())
            if group_neuron_remain == 0:
                layer_pruned.append(layer_name)
                layer_pruned2.append(layer_name)
                layer_name_list = layer_name.split(".")
                block_name = ".".join(layer_name_list[1:3])
                # print(block_name)
                block_pruned.append(block_name)
                if args.arch == 'resnet50' or args.arch == 'resnet101':
                    if 'conv1' in layer_name:
                        layer_pruned.extend([layer_name.replace('conv1', 'conv2'),
                                                layer_name.replace('conv1', 'conv3'),
                                                layer_name.replace('conv1', 'bn1'),
                                                layer_name.replace('conv1', 'bn2'),
                                                layer_name.replace('conv1', 'bn3')])
                    elif 'conv2' in layer_name:
                        layer_pruned.extend([layer_name.replace('conv2', 'conv1'),
                                                layer_name.replace('conv2', 'conv3'),
                                                layer_name.replace('conv2', 'bn1'),
                                                layer_name.replace('conv2', 'bn2'),
                                                layer_name.replace('conv2', 'bn3')])
                elif args.arch == 'resnet34':
                    if 'conv1' in layer_name:
                        layer_pruned.extend([layer_name.replace('conv1', 'conv2'),
                                                layer_name.replace('conv1', 'bn1'),
                                                layer_name.replace('conv1', 'bn2')])
            # print(layer_pruned)
            print(layer_name)
            print(group_neuron_remain)
            layer_num_dict[layer_name] = group_neuron_remain
            # layer_to_group[layer_name] = group
            total_num += group_neuron_remain
    else:
        print("Not Implemented Error.")

    print(layer_pruned2)
    block_pruned = set(block_pruned)
    print(block_pruned)
    # exit()

    if args.arch == 'resnet34':
        model = ResNet34(args.arch, layer_num_dict, num_class=num_class, add_gate=args.gate)
    elif args.arch == 'resnet50':
        model = ResNet50(args.arch, layer_num_dict, num_class=num_class, add_gate=args.gate)
        model = get_model("resnet50", "ImageNet", False, gate=False)
    elif args.arch == 'resnet101':
        model = ResNet101(args.arch, layer_num_dict, num_class=num_class, add_gate=args.gate)
    elif args.arch == 'mobilenet':
        model = mobilenet(layer_num_dict, num_class)
    elif args.arch == 'vgg16':
        model = vgg16(layer_num_dict, bias=args.enable_bias, dataset_name=args.dataset_name)
    else:
        raise NotImplementedError
    # print(model)
    # exit()
    total_num = 0
    layers = extract_layers(model, get_conv=True, get_bn=True, get_gate=args.gate)
    new_model = deepcopy(model)

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         num = module.weight.shape[0]
    #         print(f"module.{name} has {num} neurons.")

    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, nn.BatchNorm2d):
            # print(name, module)
            layer_name = name.replace("bn", "conv")
            if "downsample.1" in name:
                layer_name = name.replace("downsample.1", "downsample.0")
            module_name = "module." + layer_name
            channel_num = layer_num_dict[module_name]
            if channel_num == 0:
                use_layer = nn.Identity()
            else:
                use_layer = nn.BatchNorm2d(channel_num, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            if layer_name == 'conv1':
                setattr(new_model, name, use_layer)
                # new_model.get_submodule(name) = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            elif 'downsample' not in name:
                p1, p2, p3 = name.split('.')
                setattr(getattr(new_model, p1)[int(p2)], p3, use_layer)
                # new_model.get_submodule(p1)[int(p2)].get_submodule(p3) = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            else:
                # layer1.0.downsample.0
                p1, p2, p3, p4 = name.split('.')
                new_model.get_submodule(p1)[int(p2)].get_submodule(p3)[int(p4)] = use_layer
        # continue
        if isinstance(module, torch.nn.Conv2d):
            # print(name)
            # print(name, module.weight.shape)
            module_name = "module." + name
            if name == "layer4.2.conv3":
                conv_output_num = layer_num_dict[module_name]
            pre_group_name = pre_group[module_name]
            # print("Current Num", layer_num_dict[module_name])
            # the commented is wrong
            # if layer_masks is not None and pre_group_name in layer_masks:
            pre_active_neuron_num = None
            if layer_masks is not None and pre_group_name in groups:
                pre_active_neuron_num = get_remaining_neuron_in_group(pre_group_name, layers, layer_masks) if pre_group_name is not None else 3
                # print("Pre Active Num", pre_active_neuron_num)
            pre_active_neuron_num = 3 if pre_active_neuron_num is None else pre_active_neuron_num 
            if name == 'conv1':
                # print(new_model.get_submodule(name))
                kernel_size = new_model.get_submodule(name).kernel_size
                stride = new_model.get_submodule(name).stride
                padding = new_model.get_submodule(name).padding
                print(kernel_size, stride, padding)
                if layer_num_dict[module_name] == 0:
                    use_layer = nn.Identity()
                else:
                    use_layer = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                setattr(new_model, name, use_layer)
                # new_model.get_submodule(name) = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            # model.stage1 = [b1, b2, b3, b4]
            # setattr(model, 'stage1', [b1, b2, b3])
            elif 'downsample' not in name:
                p1, p2, p3 = name.split('.')
                # print(new_model.get_submodule(p1)[int(p2)].get_submodule(p3))
                kernel_size = new_model.get_submodule(p1)[int(p2)].get_submodule(p3).kernel_size
                stride = new_model.get_submodule(p1)[int(p2)].get_submodule(p3).stride
                padding = new_model.get_submodule(p1)[int(p2)].get_submodule(p3).padding
                # print(kernel_size, stride, padding)
                if layer_num_dict[module_name] == 0:
                    use_layer = nn.Identity()
                else:
                    use_layer = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                setattr(getattr(new_model, p1)[int(p2)], p3, use_layer)
                # new_model.get_submodule(p1)[int(p2)].get_submodule(p3) = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            else:
                # layer1.0.downsample.0
                p1, p2, p3, p4 = name.split('.')
                # print(new_model.get_submodule(p1)[int(p2)].get_submodule(p3)[int(p4)])
                kernel_size = new_model.get_submodule(p1)[int(p2)].get_submodule(p3)[int(p4)].kernel_size
                stride = new_model.get_submodule(p1)[int(p2)].get_submodule(p3)[int(p4)].stride
                padding = new_model.get_submodule(p1)[int(p2)].get_submodule(p3)[int(p4)].padding
                # print(kernel_size, stride, padding)
                if layer_num_dict[module_name] == 0:
                    use_layer = nn.Identity()
                else:
                    use_layer = nn.Conv2d(pre_active_neuron_num, layer_num_dict[module_name], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                new_model.get_submodule(p1)[int(p2)].get_submodule(p3)[int(p4)] = use_layer

            # if module.weight.numel() == 0:
            #     print(name)
    # exit()
    
    # Fully cleaned up the model
    block_pruned = sorted(list(block_pruned))[::-1]
    for pruned_block_name in block_pruned:
        print(pruned_block_name)
        p1, p2 = pruned_block_name.split(".")
        downsample_name = pruned_block_name + ".downsample"
        # print(layer2.0.downsample)
        # print(downsample_name)
        if int(p2) != 0:
            # print(p1, p2)
            del new_model.get_submodule(p1)[int(p2)]
        else:
            new_model.get_submodule(p1)[int(p2)] = new_model.get_submodule(p1)[int(p2)].downsample
        print(len(new_model.get_submodule(p1)))
        if len(new_model.get_submodule(p1)) == 0:
            setattr(new_model, p1, nn.Identity())

    setattr(new_model, "fc", nn.Linear(int(conv_output_num), 1000))

    model = new_model
    for name, param in model.named_parameters():
        total_num += param.data.numel()
    print("Total number of parameters:", total_num)
    # print(model)
    # exit()
    layer_pruned = set(layer_pruned)
    if resume_weights:
        print("=> loading weights")

        if args.arch in ['resnet50', 'resnet34', 'resnet101']:
            for layer_name in layer_pruned:
                if 'conv1' not in layer_name:
                    continue
                eps = 1e-5
                conv1_bn = layer_name.replace('conv1', 'bn1')
                out = resume_weights[conv1_bn+'.bias'].data
                conv2_layer_name = layer_name.replace('conv1', 'conv2')
                conv2_bn = conv2_layer_name.replace('conv2', 'bn2')
                layer_in_skip = conv2_layer_name
                out = torch.matmul(torch.sum(resume_weights[conv2_layer_name+'.weight'].data, dim=(2, 3)), out)
                out = (resume_weights[conv2_bn+'.weight'].data
                        * (out - resume_weights[conv2_bn+'.running_mean'].data) 
                        / (resume_weights[conv2_bn+'.running_var'].data+eps).sqrt() 
                        + resume_weights[conv2_bn+'.bias'].data)
                conv3_layer_name = layer_name.replace('conv1', 'conv3')
                if conv3_layer_name in layer_pruned:
                    layer_in_skip = conv3_layer_name
                    conv3_bn = conv3_layer_name.replace('conv3', 'bn3')
                    out = torch.matmul(torch.sum(resume_weights[conv3_layer_name+'.weight'].data, dim=(2, 3)), out)
                    out = (resume_weights[conv3_bn+'.weight'].data
                            * (out - resume_weights[conv3_bn+'.running_mean'].data)
                            / (resume_weights[conv3_bn+'.running_var'].data+eps).sqrt()
                            + resume_weights[conv3_bn+'.bias'].data)

                for group_name in layer_masks.keys():
                    if layer_in_skip in group_name:
                        mask = layer_masks[group_name]
                        resume_weights[layer_name.replace('conv1', 'add_bias')] = out[mask == 1.].view(1, -1, 1, 1)
                        break

        new_resume_weights = {}
        for k, v in resume_weights.items():
            if 'add_bias' in k:
                new_resume_weights[k.replace('module.', '')] = v
                continue
            layer_name = '.'.join((k.split('.')[:-1]))
            if 'bn' in layer_name:  # deal with bn layers
                layer_name = layer_name.replace('bn', 'conv')
                layer_name = layer_name.replace('conv_conv', 'conv_bn')  # specially for mobilenet
            elif 'downsample.1' in layer_name:  # deal with bn layer in skip connection
                layer_name = layer_name.replace('downsample.1', 'downsample.0')
            if layer_masks is not None and len(v.size()) > 0 and 'fc' not in layer_name and 'linear' not in layer_name:
                mask = layer_masks[layer_to_group[layer_name]]
                if layer_name not in layer_pruned:
                    new_resume_weights[k.replace('module.', '')] = v[mask == 1.]
                # remove the corresponding params in the next layer(s)
                if ('conv' in k or 'downsample.0' in k) and '.bn' not in k:
                    next_layers = adjacency[layer_name] if layer_name in adjacency else []
                    for next_layer in next_layers:
                        if next_layer == 'module.classifier.linear1' and 'bias' not in k:
                            next_layer_weight_name = next_layer+'.weight'
                            linear_mask = torch.zeros((mask.size(0), 7, 7))
                            linear_mask[mask.bool(), :, :] = 1
                            linear_mask = torch.flatten(linear_mask)
                            resume_weights[next_layer_weight_name] = resume_weights[next_layer_weight_name][:, linear_mask == 1]
                        elif 'bias' not in k:
                            next_layer_weight_name = next_layer.replace('module.', '')+'.weight'
                            if next_layer_weight_name in new_resume_weights:
                                if resume_weights[next_layer_weight_name].size(1) != 1:
                                    new_resume_weights[next_layer_weight_name] = new_resume_weights[next_layer_weight_name][:, mask == 1., :, :]
                            else:
                                next_layer_weight_name = next_layer + '.weight'
                                if resume_weights[next_layer_weight_name].size(1) != 1:
                                    resume_weights[next_layer_weight_name] = resume_weights[next_layer_weight_name][:, mask == 1., ...]
                        else:
                            continue
            else:
                if layer_name not in layer_pruned:
                    new_resume_weights[k.replace('module.', '')] = v
        model.load_state_dict(new_resume_weights)

    return model


if __name__ == '__main__':
    args = args_parser()
    model = main(args)
    if '.tar' in args.pre_trained:
        torch.save(model.state_dict(), args.pre_trained.replace('.tar', '_clean.pth'))
    else:
        torch.save(model.state_dict(), args.pre_trained.replace('.pth', '_clean.pth'))
