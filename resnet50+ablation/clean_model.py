import os
import argparse
import pickle as pkl
import json

import torch
#from thop import profile, clever_format
from models.resnet_pruned import ResNet34, ResNet50, ResNet101
from models.mobilenet_pruned import mobilenet
from models.vgg_pruned import vgg16


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
    if args.mask is not None:
        with open(args.mask, 'rb') as f:
            group_mask = pkl.load(f)
        # print(group_mask)
        for group, mask in group_mask.items():
            group_neuron_remain = int(torch.sum(mask).item())
            if group_neuron_remain == 0:
                for layer_name in group:
                    layer_pruned.append(layer_name)
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
            for layer_name in group:
                layer_num_dict[layer_name] = group_neuron_remain
                layer_to_group[layer_name] = group
            total_num += group_neuron_remain*len(group)
    else:
        if args.layer_size_file is None:
            group_mask = None
            for k, v in resume_weights.items():
                layer_name = '.'.join((k.split('.')[:-1]))
                if layer_name not in layer_num_dict and ('conv' in layer_name or 'downsample.0' in layer_name):
                    try:
                        layer_num_dict[layer_name] = int(v.size(0))
                        total_num += layer_num_dict[layer_name]
                    except:
                        continue
        else:
            group_mask = None
            with open(args.layer_size_file, 'r') as f:
                layer_num_dict = json.load(f)

    if args.arch == 'resnet34':
        model = ResNet34(args.arch, layer_num_dict, num_class=num_class, add_gate=args.gate)
    elif args.arch == 'resnet50':
        model = ResNet50(args.arch, layer_num_dict, num_class=num_class, add_gate=args.gate)
    elif args.arch == 'resnet101':
        model = ResNet101(args.arch, layer_num_dict, num_class=num_class, add_gate=args.gate)
    elif args.arch == 'mobilenet':
        model = mobilenet(layer_num_dict, num_class)
    elif args.arch == 'vgg16':
        model = vgg16(layer_num_dict, bias=args.enable_bias, dataset_name=args.dataset_name)
    else:
        raise NotImplementedError

    total_num = 0
    for name, param in model.named_parameters():
        total_num += param.data.numel()
    print("Total number of parameters:", total_num)
    
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

                for group_name in group_mask.keys():
                    if layer_in_skip in group_name:
                        mask = group_mask[group_name]
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
            if group_mask is not None and len(v.size()) > 0 and 'fc' not in layer_name and 'linear' not in layer_name:
                mask = group_mask[layer_to_group[layer_name]]
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

    print(model)
    # exit()
    return model


if __name__ == '__main__':
    args = args_parser()
    model = main(args)
    if '.tar' in args.pre_trained:
        torch.save(model.state_dict(), args.pre_trained.replace('.tar', '_clean.pth'))
    else:
        torch.save(model.state_dict(), args.pre_trained.replace('.pth', '_clean.pth'))
