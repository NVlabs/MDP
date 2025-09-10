from ortools.algorithms import pywrapknapsack_solver
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import shutil
import time
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from utils.utils import args_parser, extract_layers, is_conv
from utils.smoothing import LabelSmoothing
from utils.mixup import NLLMultiLabelSmooth, MixUpWrapper
from utils.dataloaders import get_pytorch_train_loader, get_pytorch_val_loader

from models.models import get_model
# from training import *
from utils.lr_schedule import *
from prune.prune_config import PruneConfigReader
from prune.pruning import set_prune_target


def main(args):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    args.cuda = True

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print("Warning: simulated batch size {} is not divisible by actual batch size {}".format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size/tbs)
        print("BSM: {}".format(batch_size_multiplier))

    resume_weights = None
    resume_optimizer = None
    if args.pre_trained:
        if os.path.isfile(args.pre_trained):
            if args.pre_trained.endswith('.pth'):
                print("=> loading checkpoint weights from '{}'".format(args.pre_trained))
                checkpoint = torch.load(args.pre_trained, map_location=lambda storage, loc: storage)
                resume_weights = checkpoint
            elif args.pre_trained.endswith('.tar'):
                print("=> loading checkpoint weights and optimizer from '{}'".format(args.pre_trained))
                checkpoint = torch.load(args.pre_trained, map_location=lambda storage, loc: storage)
                resume_weights = checkpoint['model_state_dict']
                resume_optimizer = checkpoint['optimizer_state_dict']
            else:
                print("=> unsupported file for loading at '{}'".format(args.pre_trained))
        else:
            print("=> no checkpoint weights found at '{}'".format(args.pre_trained))

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    model = get_model(args.arch, args.dataset_name, args.enable_bias, args.gate, args.collapse, args.collapse2)
    if resume_weights:
        print("=> loading weights")
        new_resume_weights = {k.replace('module.', ''): v for k, v in resume_weights.items()}
        model.load_state_dict(new_resume_weights)
    if args.cuda:
        model = model.cuda()
    if args.fp16:
        model = network_to_half(model)
    print(model)
    criterion = loss()
    if args.cuda:
        criterion = criterion.cuda()

    if args.dataset_name == 'ImageNet':
        class_num = 1000
    elif args.dataset_name == 'CIFAR10':
        class_num = 10
    else:
        raise NotImplementedError('Dataset {} has not been supported! '.format(args.dataset_name))
    # Create data loaders and optimizers as needed
    get_train_loader = get_pytorch_train_loader
    get_val_loader = get_pytorch_val_loader

    train_loader, train_loader_len = get_train_loader(args.data_root, args.batch_size, class_num, args.mixup > 0.0,
                                                      workers=args.workers, fp16=(args.fp16 or args.amp), dataset=args.dataset_name)

    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, class_num, train_loader)

    val_loader, val_loader_len = get_val_loader(args.data_root, args.batch_size, class_num, False,
                                                workers=args.workers, fp16=(args.fp16 or args.amp), dataset=args.dataset_name)

    if not args.gate:
        parameters_for_optimizer = list(model.named_parameters())
    else:
        parameters_for_optimizer = []
        for name, m in model.named_parameters():
            # gate parameters are not optimized
            if "gate" not in name:
                parameters_for_optimizer.append((name, m))
            else:
                print("skipping parameter", name, "shape:", m.shape)
    total_size_params = sum([np.prod(par[1].shape) for par in parameters_for_optimizer])
    print("Total number of trainable parameters: ", total_size_params)

    optimizer = get_optimizer(parameters_for_optimizer,
                              args.fp16,
                              args.learning_rate, args.momentum, args.weight_decay,
                              nesterov=args.nesterov,
                              bn_weight_decay=args.bn_weight_decay,
                              static_loss_scale=args.static_loss_scale,
                              dynamic_loss_scale=args.dynamic_loss_scale)
    if resume_optimizer:
        print("=> loading optimizer")
        optimizer.load_state_dict(resume_optimizer)

    lr_scheduler = None
    if args.lr_schedule == 'step':
        lr_scheduler = lr_step_policy(args.learning_rate, [30,60,80], 0.1, args.warmup)
    elif args.lr_schedule == 'step_prune':
        lr_scheduler = lr_step_policy(args.learning_rate, [10,20,30], 0.1, args.warmup)
    elif args.lr_schedule == 'cosine':
        lr_scheduler = lr_cosine_policy(args.learning_rate, args.warmup, args.epochs)
    elif args.lr_schedule == 'linear':
        lr_scheduler = lr_linear_policy(args.learning_rate, args.warmup, args.epochs)

    if args.amp:
        args.dynamic_loss_scale = True
        amp.register_float_function(torch, 'batch_norm')
        model, optimizer = amp.initialize(
                model, optimizer,
                opt_level="O1",
                loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale)

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    else:
        model = torch.nn.DataParallel(model)

    ################
    # set up output directory
    ################
    verbose = (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    train_writer = None
    if verbose:
        train_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_log'))

    layers = extract_layers(model, get_conv=True, get_bn=True, get_gate=args.gate)
    assert (args.gate == (args.reg_conf is not None) or not args.gate)  # if gate is added, specify the structure
    if args.reg_conf is not None:
        config_reader = PruneConfigReader()
        config_reader.set_prune_setting(args.reg_conf)
        layer_structure = config_reader.get_layer_structure()  # conv_bn, conv_gate, groups
    else:
        conv_bn, conv_gate = None, None
        groups = [tuple([layer_name]) for layer_name, layer in layers.items() if is_conv(layer)]
        layer_structure = conv_bn, conv_gate, groups
    if not args.no_prune:
        total_conv_num = count_non_zero_neurons(layers)
        assert args.prune_mode in ['exp', 'exp2', 'linear']
        if args.prune_per_epoch:
            scheduled_num = set_prune_target(total_conv_num, args.prune_ratio,
                                             args.reg_start_epoch, args.reg_end_epoch,
                                             args.prune_mode)
        else:
            scheduled_num = set_prune_target(total_conv_num, args.prune_ratio,
                                             0, 30,
                                             args.prune_mode)
        prune_start = args.reg_start_epoch
        prune_end = args.reg_end_epoch
    else:
        scheduled_num, prune_start, prune_end = None, -1, -1
    if not args.gate:
        conv_bn, conv_gate, groups = layer_structure
        layer_structure = conv_bn, None, groups
    if args.baseline_file is not None and os.path.exists(args.baseline_file):
        baseline_acc = np.loadtxt(args.baseline_file)
    else:
        baseline_acc = None

    prec1, prec5 = validate(val_loader, model, criterion)
    print('top1/top5 acc right after loading: {}, {}'.format(prec1, prec5))
    best_prec1 = train_loop(args,
                    model,
                    criterion,
                    optimizer,
                    lr_scheduler,
                    train_loader,
                    train_loader_len,
                    val_loader,
                    args.epochs,
                    args.fp16,
                    use_amp=args.amp,
                    batch_size_multiplier=batch_size_multiplier,
                    start_epoch=args.start_epoch,
                    best_prec1=best_prec1,
                    output_dir=args.output_dir,
                    train_log_freq=args.train_log_freq,
                    ckpt_freq=args.ckpt_freq,
                    verbose=verbose,
                    train_writer=train_writer,
                    layers=layers,
                    layer_structure=layer_structure,
                    prune_target=scheduled_num,
                    prune_start=prune_start,
                    prune_end=prune_end,
                    method=args.method,
                    prune_per_epoch=args.prune_per_epoch,
                    baseline_acc=baseline_acc,
                    pre_mask=args.mask,
                    wrap_solver=pywrapknapsack_solver,
                    clipping=args.clip)

    print("Experiment ended")
    print("Best top1 accuracy: {}".format(best_prec1))

    total_conv_num = count_non_zero_neurons(layers, verbose=True)
    print('The final network has {} neurons in total.'.format(total_conv_num))


if __name__ == '__main__':
    parser = args_parser()
    args = parser.parse_args()
    cudnn.benchmark = True
    if args.mgp:
        from training_mgp import *
    elif args.mdlm:
        from training_mdlm import *
    elif args.mdp:
        from training_mdp import *
    else:
        from training import *
    main(args)
