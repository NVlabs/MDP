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
    with open("/home/xinglongs/pascal-detection/group_mask_ssd.pkl", 'rb') as f:
            group_mask = pkl.load(f)
            print(group_mask)
            layer_num_dict = {}
            layer_to_group = {}
            total_num = 0
            layer_pruned = []
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
    print(layer_num_dict)
    backbone = ResNet50("resnet50", layer_num_dict, num_class=21, add_gate=False)
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
    exit()
    model = get_clean_model(args)

    device = torch.device(args.gpu)

    model.eval()
    model.to(device)

    # get_val_loader = get_pytorch_val_loader
    # val_loader, val_loader_len = get_val_loader(args.data_root, args.batch_size, 1000, False,
    #                                             workers=args.workers, fp16=False)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times = []
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1,3,1,1).float()
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1,3,1,1).float()
    for i in range(60):
        # 224 -> 150
        input = torch.randint(low=0, high=255, size = (args.batch_size, 3, 224, 224)).float()
        input = input.sub_(mean).div_(std)
        input = input.to(device)
        start_evt.record()
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
    print('Infer time (ms/image)', np.mean(times)/args.batch_size)
    print('FPS:', args.batch_size*1e+3 / np.mean(times))

    if args.dataset_name == 'ImageNet':
        input = torch.randn(1, 3, 224, 224)
    elif args.dataset_name == 'CIFAR10':
        input = torch.randn(1, 3, 32, 32)
    else:
        raise NotImplementedError
    flops = model_summary(model, input.cuda())
    print('MACs(G): {:.3f}'.format(flops / 1e9))


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
   main()
