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

# from clean_model import main as get_clean_model
# from utils.dataloaders import get_pytorch_val_loader
from utils.model_summary import model_summary

def main():
    parser = args_parser()
    args = parser.parse_args()
    if args.mgp:
        from clean_model_mgp import main as get_clean_model
    else:
        from clean_model import main as get_clean_model
    cudnn.benchmark = True
    cudnn.deterministic = True

    args.gpu = 0
    args.cuda = True

    torch.set_grad_enabled(False)

    model = get_clean_model(args)

    device = torch.device(args.gpu)

    model.eval()
    model.to(device)

    # Generate Onnx
    # dummy_input = torch.randn(args.batch_size, 3, 224, 224, device="cuda")
    # model = torchvision.models.alexnet(pretrained=True).cuda()

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    # input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    # output_names = [ "output1" ]
    # torch.onnx.export(model, dummy_input, "halp70_latv2_hahp_dla_oneshot.onnx", verbose=True)
    # exit()
    # torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

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
    parser.add_argument('--mgp', action='store_true', help='Use Multi-Granularity Pruning.')

    return parser



if __name__ == '__main__':
    main()
