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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import argparse
from datetime import datetime
import pytz
from shutil import copyfile

# Model parameters
# Not too many here since the SSD300 has a very specific structure
model_names = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', \
               'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
n_classes = len(label_map)  # number of different types of objects


def main(gpu, args):
    """
    Training.
    """
    global label_map
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.gpus, rank=gpu)
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    torch.cuda.set_device(gpu)

    # Initialize model or load checkpoint
    model = SSD(model=args.model, backbone=args.arch, n_classes=n_classes, pretrained=args.pretrained, batch_norm=not args.disable_batch_norm, no_gating=not args.enable_gating, init_method=args.init_method)
    if gpu == 0:
        print(model)
    resume_weights, resume_optimizer = None, None
    if args.load_ckpt is not None:
        checkpoint = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage)
        resume_weights = checkpoint['model_state_dict']
        resume_optimizer = checkpoint['optimizer_state_dict']
        print('====> loading weights from {}'.format(args.load_ckpt))
        new_resume_weights = {k.replace('module.', ''): v for k, v in resume_weights.items()}
        model.load_state_dict(new_resume_weights)

    model.cuda(gpu)
    cudnn.benchmark = True

    if args.model=='SSD300':
        input_size = (300, 300)
    else:
        input_size = (512, 512)
    # Custom dataloaders
    train_dataset = PascalVOCDataset(args.data_dir,
                                     split='train',
                                     keep_difficult=(not args.disable_difficult), input_size=input_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.gpus, rank=gpu)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size/args.gpus), shuffle=False, sampler=train_sampler,
                                               collate_fn=train_dataset.collate_fn, num_workers=int(args.workers/args.gpus),
                                               pin_memory=True)  # note that we're passing the collate function here

    if args.coarse_pruning:
        from optimizer import NetOptimizer
        from prune.prune_config import PruneConfigReader
    else:
        from optimizer_fine import NetOptimizer
        from prune.prune_config_fine import PruneConfigReader

    config_reader = PruneConfigReader(gpu==0)
    config_reader.set_prune_setting(args.reg_conf)
    layer_structure = config_reader.get_layer_structure()

    net_optimizer = NetOptimizer(model, len(train_dataset), train_loader, args, layer_structure, gpu)
    if resume_optimizer is not None:
        print('====> loading optimizer')
        net_optimizer.optimizer.load_state_dict(resume_optimizer)

    net_optimizer.optimize_net()
    # END


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--data-dir',
                        help='JSON directory for training/test sets')
    parser.add_argument('--checkpoint', metavar='F',
                        help='File path of checkpoint')
    parser.add_argument('--disable-difficult', action='store_true')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--reg-conf', type=str, default='configs_fine/resnet34_ordered.json',
                        help='The json file defining the regularization configuration.')
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='The number of examples in one training batch (default: 128)')
    parser.add_argument('--enable-bias', type=bool, default=False,
                        help='Whether to enable the bias term in the convolution layers')
    parser.add_argument('--epochs', type=int, default=800,
                         help='Number of epochs to train')
    parser.add_argument('--lrs', nargs='+', type=float,
                        default=[4e-4, 8e-3, 8e-3, 3e-3, 3e-3, 1e-3, 1e-3, 4e-4, 4e-4, 4e-5, 4e-5],
                        help='learning rates for the optimizer')
    parser.add_argument('--lr-schedule', nargs='+', type=int,
                        default=[50, 600, 601, 700, 701, 740, 741, 770, 771, 800],
#                         default=[50, 700, 701, 800, 801, 840, 841, 870, 871, 900],
                        # default=[50, 700, 701, 850, 851, 890, 891, 920, 921, 1000],
                        help='learning rates for the optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--mu', default=6e-4, type=float, help='the scalar for latency aware importance')
    parser.add_argument('--regrow-ratio', default=0.75, type=float, help='Latency Regrow Ratio')
    
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov momentum, default: false)')
    parser.add_argument('--weight-decay', type=float, default=2e-3,
                        help='weight decay factor for the optimizer')
    parser.add_argument('--disable-batch-norm', action='store_true',
                        help='Disable batch norm for additional layers.')
    parser.add_argument('--bn-weight-decay', default=0.0, type=float,
                        help='weight decay on BN (default: 0.0)')
    parser.add_argument('--grad-clip', type=float,
                        help='Gradient clipping to evade nans')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet34)')
    parser.add_argument('--prune-method', default='linear',
                        choices=['linear', 'uniform', 'exp', 'exp2'],
                        help='the method used for pruning.')
    parser.add_argument('--prune-ratio', type=float, default=0,
                        help='the ratio of neurons going to be pruned.')
    parser.add_argument('--prune-start', type=int, default=None,
                        help='the epoch to start pruning.')
    parser.add_argument('--prune-end', type=int, default=None,
                        help='the epoch to end pruning.')
    parser.add_argument('--prune-interval', type=int, default=1,
                        help='the interval for pruning. By default, pruning is done once every epoch.')
    parser.add_argument('-m', '--method', type=int, default=1,
                        help='the method to rank the neurons during the pruning')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--enable-gating', action='store_true')
    parser.add_argument('--gpus', default=2, type=int,
                        help='number of gpus per node')
    # O1 before
    parser.add_argument('--opt-level', default='O1',
                        help='Optimization level for AMP')
    parser.add_argument('--checkpoint-all', action='store_true')
    parser.add_argument('--model', type=str, default='SSD300',
                        choices=['SSD300', 'SSD512', 'ssd300', 'ssd512'],
                        help='Which SSD model to use (default: SSD300)')
    parser.add_argument('--no-prune', action='store_true', help='Perform pruning.')
    parser.add_argument('--disable-linear-lr', action='store_true',
                        help='Disable linear warmup for learning rates. Linear-lr linearly connects all learning rates according to the lr schedule.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained models for backbone.')
    parser.add_argument('--prune-per-epoch', action='store_true',
                        help='if true, perform pruning at the end of each epoch. '
                             'Otherwise, prune every 40 iterations and finishes in 30 steps for imagenet')
    parser.add_argument('--init-method', type=str, default='xavier',
                        choices=['xavier', 'kaiming-fan-out', 'kaiming-fan-in'],
                        help='Which initialization method to use')
    parser.add_argument('--coarse-pruning', action='store_true',
                        help='Use channel pruning instead of weight pruning.')
    parser.add_argument('--target-latency', default=None, type=float, help='the targeted latency')
    parser.add_argument('--two-to-four', action='store_true',
                        help='Use 2:4 pruning. This fixes prune ratio to 0.5 and applies one-shot pruning at epoch prune_start.')
    parser.add_argument('--two-to-four-spread', action='store_true',
                        help='Use 2:4 pruning, spread out. This fixes prune ratio to 0.5 and applies pruning starting at epoch prune_start.')
    parser.add_argument('--two-to-four-spread-reverse', action='store_true',
                        help='Start from the most behind layer and go forward to prune.')
    parser.add_argument('--timezone', type=str, default='US/Eastern',
                        help='Set timezone to save results')
    parser.add_argument('--output-dir', type=str, default='/result',
                        help='Set the output directory to save files and Tensorboards.')
    parser.add_argument('--load-ckpt', type=str, default=None)
    parser.add_argument('--backbone-only', action='store_true', default=False,
                        help='only prune backbone')
    args = parser.parse_args()
    
    assert args.coarse_pruning or args.method in [1, 2]
    assert len(args.lr_schedule)+1 == len(args.lrs)
    assert args.disable_linear_lr or args.lr_schedule[-1] == args.epochs
    args.model = args.model.upper()
    assert args.model in ['SSD300', 'SSD512']
    args.decay_rates = [args.lrs[i]/args.lrs[i-1] for i in range(1, len(args.lrs))]
    args.lrs = [(args.batch_size/128) * x for x in args.lrs]
    args.lr = args.lrs[0]

    tz = pytz.timezone(args.timezone)
    now = str(datetime.now(tz)).split('.')[0]
    curr_time = now.replace(' ', '_').replace(':', '-')
    print(curr_time)

    args.output_dir = os.path.join(args.output_dir, curr_time)
    if not os.path.exists(args.output_dir):
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
    
    args.tensorboard_dir = os.path.join(args.output_dir, 'runs')
    if not os.path.exists(args.tensorboard_dir):
        path = Path(args.tensorboard_dir)
        path.mkdir(parents=True, exist_ok=True)
        
    args.checkpoint_dir = os.path.join(args.output_dir, 'VOC_checkpoints')
    if not os.path.exists(args.checkpoint_dir):
        path = Path(args.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12357'
    mp.spawn(main, nprocs=args.gpus, args=(args,))

