import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from ssd_models.models import SSD
from losses import MultiBoxLoss
from datasets import PascalVOCDataset
# from clean_model import main as get_clean_model
# from utils.dataloaders import get_pytorch_val_loader
from utils.model_summary import model_summary
import time, sys, os
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from ssd_models.models import SSD
from losses import MultiBoxLoss
from datasets import PascalVOCDataset
from utils.utils import *
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import argparse
from datetime import datetime
import pytz
from shutil import copyfile
import pickle as pkl
from prune.hahp_func import *
from resnet_pruned import ResNet34, ResNet50, ResNet101
# Model parameters
# Not too many here since the SSD300 has a very specific structure
model_names = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', \
               'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
n_classes = len(label_map)  # number of different types of objects

def main():
    parser = args_parser()
    args = parser.parse_args()
    cudnn.benchmark = True
    cudnn.deterministic = True

    args.gpu = 0
    args.cuda = True

    torch.set_grad_enabled(False)
    model = SSD("SSD512", backbone="resnet50", n_classes=n_classes, pretrained=False, batch_norm=True, no_gating=True, init_method="xavier")
    # with open("/home/xinglongs/pascal-detection/test.pkl", 'rb') as f:
    # with open("/home/xinglongs/pascal-detection/test.pkl", 'rb') as f:
    # with open("/home/xinglongs/results/5008343/2023-08-16_15-28-10/group_mask.pkl", 'rb') as f:
    with open("/home/xinglongs/results/summary/ssd/ssd_halp2_hahp_oneshot_60/layer_masks.pkl", 'rb') as f:
    # with open("/home/xinglongs/pascal-detection/group_mask_2.pkl", 'rb') as f:
        layer_masks = pkl.load(f)
    # print(group_mask)
    # group_mask = {k.replace("f_0.", ""):v for k, v in group_mask.items()}
    layer_num_dict = {}
    layer_to_group = {}
    total_num = 0
    layer_pruned = []
    layer_pruned2 = []
    block_pruned = []

    print(layer_masks.keys())
    for layer_name, mask in layer_masks.items():
        group_neuron_remain = int(torch.sum(mask).item())
        if group_neuron_remain == 0:
            layer_pruned.append(layer_name)
            layer_pruned2.append(layer_name)
            layer_name_list = layer_name.replace('f_0.', '').split(".")
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

    print(layer_pruned2)
    block_pruned = set(block_pruned)
    print(block_pruned)

    # print(layer_num_dict)
    layer_num_dict = {k.replace("f_0.", ""):v for k, v in layer_num_dict.items()}
    layer_masks = {k.replace("f_0.", ""):v for k, v in layer_masks.items()}
    block_pruned = [x.replace("f_0.", "") for x in block_pruned]
    print(layer_num_dict.keys())
    # backbone = ResNet50("resnet50", layer_num_dict, num_class=21, add_gate=False)
    from ssd_models.resnet_feature_extractor import ResNet50
    backbone, out_channels = ResNet50(21, pretrained=False)
    print(backbone)

    from copy import deepcopy
    new_model = deepcopy(backbone)
    layers = extract_layers(backbone, get_conv=True, get_bn=True, get_gate=args.gate)
    # print(layers.keys())
    # exit()
    from prune.prune_config import PruneConfigReader
    config_reader = PruneConfigReader()
    config_reader.set_prune_setting("resnet50.json")
    layer_structure = config_reader.get_layer_structure()  # conv_bn, conv_gate, groups
    conv_bn, conv_gate, groups = layer_structure
    from prune.prune_config_with_structure import PruneConfigReader
    config_reader = PruneConfigReader()
    # config_reader.set_prune_setting("configs/resnet50_structure.json")
    config_reader.set_prune_setting("resnet50_structure.json")
    _, _, _, pre_group, aft_group_list = config_reader.get_layer_structure()

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         num = module.weight.shape[0]
    #         print(f"module.{name} has {num} neurons.")
    # ******************************************************************************************************************************
    for name, module in backbone.named_modules():
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

    # setattr(new_model, "fc", nn.Linear(int(conv_output_num), 1000))


    # ******************************************************************************************************************************
    # exit()
    # checkpoint = torch.load(checkpoint)
    # model_state_dict = OrderedDict()
    # if 'model_state_dict' in checkpoint:
    #     for k, v in checkpoint['model_state_dict'].items():
    #         model_state_dict[k.replace("features_0", "f_0")] = v
    # else:
    #     for k, v in checkpoint.items():
    #         model_state_dict[k.replace("features_0", "f_0")] = v

    # # layers = extract_layers(model, get_conv=True, get_bn=True, get_gate=False)
    # # config_reader = PruneConfigReader(gpu==0)
    # # config_reader.set_prune_setting(args.reg_conf)
    # # layer_structure = config_reader.get_layer_structure()
    # for k, v in model_state_dict.items():
    #     if '_orig' in k:
    #         new_k = k.replace('_orig', '')
    #         # print(model_state_dict[k].shape)
    #         model_state_dict[k] = model_state_dict[k] * model_state_dict[f"{new_k}_mask"]
    #         # print(model_state_dict[k].shape)

    # model_state_dict = {k.replace('_orig', ''): v for k, v in model_state_dict.items() if "mask" not in k}
    # for key in model_state_dict:
    #     if "_mask" in key:
    #         del model_state_dict[key]
    # print(model_state_dict)

    device = torch.device(args.gpu)

    # model.f_0 = backbone
    model.f_0 = new_model
    model.eval()
    model.to(device)
    # backbone.eval()
    # backbone.to(device)

    # get_val_loader = get_pytorch_val_loader
    # val_loader, val_loader_len = get_val_loader(args.data_root, args.batch_size, 1000, False,
    #                                             workers=args.workers, fp16=False)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times = []
    # mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1,3,1,1).float()
    # std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1,3,1,1).float()
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1,3,1,1).float()
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1,3,1,1).float()
    # input = torch.randint(low=0, high=255, size = (args.batch_size, 3, 224, 224)).float()
    # input = input.sub_(mean).div_(std)
    # input = input.to(device)
    
    # start_evt.record()
    # # output = model(input)
    # output = backbone(input)
    # print(output)
    batch_size = 1
    for i in range(60):
        # 224 -> 150
        input = torch.randint(low=0, high=255, size = (batch_size, 3, 512, 512)).float()
        # input = input.sub_(mean).div_(std)
        input = input.to(device)
        start_evt.record()
        # print(input.shape)
        output = model(input)
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_time = start_evt.elapsed_time(end_evt)
        if i < 20:
            continue
        times.append(elapsed_time)
        if i >= 30:
            break
    print(times)
    print('Infer time (ms/image)', np.mean(times)/batch_size)
    print('FPS:', batch_size / np.mean(times) * 1e+3)

    # if args.dataset_name == 'ImageNet':
    #     input = torch.randn(1, 3, 224, 224)
    # elif args.dataset_name == 'CIFAR10':
    #     input = torch.randn(1, 3, 32, 32)
    # else:
    #     raise NotImplementedError
    input = torch.randint(low=0, high=255, size = (1, 3, 512, 512)).float()
    flops = model_summary(model, input.cuda())
    print('MACs(G): {:.3f}'.format(flops / 1e9))
    print(layer_num_dict)

def args_parser():
    parser = argparse.ArgumentParser(description='Script for testing ideas of pruning in pytorch')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('--enable-bias', type=bool, default=False,
                        help='Whether to enable the bias term in the convolution layers')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate layers in the network')
    parser.add_argument('--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The number of examples in one training batch')
    parser.add_argument('--dataset-name', type=str, default='ImageNet',
                        help='The name of the dataset')
    parser.add_argument('--data-root', type=str, default='/mnt/data/',
                        help='The root directory of the dataset')

    parser.add_argument('--pre-trained', type=str, default=None,
                        help='the path of a pre-trained model if want to start from some checkpoint')
    parser.add_argument('--mask', type=str, default=None,
                        help='the mask file')
    parser.add_argument('--layer-size-file', type=str, default=None, help='file saving size of each layer')

    return parser

if __name__ == '__main__':
    main()
