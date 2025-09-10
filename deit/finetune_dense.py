import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ApexScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, prune_with_Taylor, finetune_one_epoch
from losses import DistillationLoss
from samplers import RASampler
import models
import utils

from torch.utils.tensorboard import SummaryWriter

from pruning_core.pruning_utils import prepare_logging, get_lr, lr_cosine_policy, initialize_pruning_engine
from model_pruning import create_pruning_structure_vit, enable_pruning

#from apex.contrib.sparsity import ASP
from einops import rearrange

#try:
#    from apex import amp
#    from apex.parallel import DistributedDataParallel as ApexDDP
#    from apex.parallel import convert_syncbn_model

#    has_apex = True
#except ImportError:
has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained model')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='use pretrained model')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--asp', action='store_true', default=False,
                        help='Apply ampere sparsity before finetuning')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument(
        "--hidden_loss_coeff",
        type=float,
        default=3e-2,
        required=False,
        help=
        "loss for cross feature map regularization on layernorms"
    )

    parser.add_argument(
        "--original_loss_coeff",
        type=float,
        default=0.0,
        required=False,
        help=
        "original QA loss for pruning"
    )

    parser.add_argument(
        "--kl_loss_coeff",
        type=float,
        default=1000.0,
        required=False,
        help=
        "KL loss for pruning"
    )
    
    return parser


class Knowledge_Distillation_Loss(torch.nn.Module):
    def __init__(self, T = 3):
        super(Knowledge_Distillation_Loss, self).__init__()
        self.KLdiv = torch.nn.KLDivLoss()
        self.T = T

    def get_knowledge_distillation_loss(self, output_student, output_teacher):
        loss_kl = self.KLdiv(torch.nn.functional.log_softmax(output_student / self.T, dim=1), torch.nn.functional.softmax(output_teacher / self.T, dim=1))

        loss = loss_kl
        return loss



def main(args):
    utils.init_distributed_mode(args)

    print(args)
    
    if args.scratch:
        args.output_dir = args.finetune+'/scratch_'+str(args.epochs)+'_lr_'+str(args.lr)+'_a_'+str(args.distillation_alpha)+'_T_'+str(args.distillation_tau)
    else:
        args.output_dir = args.finetune+'/ft_dense_'+str(args.epochs)+'_lr_'+str(args.lr)+'_a_'+str(args.distillation_alpha)+'_T_'+str(args.distillation_tau)
    if args.asp:
        args.output_dir += '_asp'
    if args.kl_loss_coeff==0.:
        args.output_dir += '_CNN_only'
    if args.resume:
        args.finetune = args.output_dir+'/ft_checkpoint.pth'
    else:
        args.finetune = args.finetune+'/pruned_checkpoint.pth'
    if utils.is_main_process():
        writer = SummaryWriter(args.output_dir)
    #if args.distillation_type != 'none' and args.finetune and not args.eval:
    #    raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Number of iterations per epoch.
    args.iters_per_epoch = len(data_loader_train)

    # The total number of iterations.
    args.train_iters = args.epochs * args.iters_per_epoch


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        print('nomix1')

    print(f"Creating model: {args.model}")
    teacher_model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    student_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    teacher_model.to(device)
    student_model.to(device)
    
    output_dir = Path(args.output_dir)
    if args.finetune and not args.resume:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        student_model.load_state_dict(checkpoint['model'])
        total_epoch = checkpoint['epoch'] + 1
        if args.scratch:
            total_epoch = 0
        #utils.print_model_parameters(student_model)
        head = []
        QK = []
        V = []
        MLP = []
        for name, p in student_model.named_parameters():
            tensor = p.data.cpu().numpy()
            mask = np.where(abs(tensor)==0., 0., 1.)
            if 'patch_embed' in name and 'weight' in name:
                EMB_mask = np.sum(mask, axis=(1,2,3))
                EMB = np.nonzero(np.sum(mask, axis=(1,2,3)))[0].size
            elif 'attn.qkv.Q' in name and 'weight' in name:
                h = np.nonzero(np.sum(mask, axis=(0,1)))[0].size
                q = np.nonzero(np.sum(mask, axis=(1,2)))[0].size
                head.append(h)
                QK.append(q)
            elif 'attn.qkv.V' in name and 'weight' in name:
                v = np.nonzero(np.sum(mask, axis=(1,2)))[0].size
                V.append(v)
            elif '.mlp.fc1' in name and 'weight' in name:
                m = np.nonzero(np.sum(mask, axis=(1)))[0].size
                MLP.append(m)
        print(EMB,head,QK,V,MLP)
        dim_dict = {'EMB':EMB,'head':head,'QK':QK,'V':V,'MLP':MLP}
    elif args.resume:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        dim_dict = checkpoint['dim']
    
    print(f"Creating dense model: {args.model}")
    dense_model = create_model(
        args.model+'_small',
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        EMB=dim_dict['EMB'],
        QK=dim_dict['QK'],
        V=dim_dict['V'],
        MLP=dim_dict['MLP'],
        head=dim_dict['head'],
    )
    dense_model.to(device) 
    #utils.print_model_parameters(dense_model)
    
    if not args.resume and not args.scratch:
        ## Move all variables from sparse to dense model
        dense_model.cls_token.data = student_model.cls_token.data[:,:,EMB_mask!=0]
        dense_model.dist_token.data = student_model.dist_token.data[:,:,EMB_mask!=0]
        dense_model.pos_embed.data = student_model.pos_embed.data[:,:,EMB_mask!=0]
        dense_model.patch_embed.proj.weight.data = student_model.patch_embed.proj.weight.data[EMB_mask!=0,:,:,:]
        dense_model.patch_embed.proj.bias.data = student_model.patch_embed.proj.bias.data[EMB_mask!=0]
        
        for blk_dense, blk in zip(dense_model.blocks, student_model.blocks):
            blk_dense.norm1.weight.data = blk.norm1.weight.data[EMB_mask!=0]
            blk_dense.norm1.bias.data = blk.norm1.bias.data[EMB_mask!=0]
            blk_dense.norm2.weight.data = blk.norm2.weight.data[EMB_mask!=0]
            blk_dense.norm2.bias.data = blk.norm2.bias.data[EMB_mask!=0]
            blk_dense.attn.qkv.token_mask.data = blk.attn.qkv.token_mask.data
            
            MLP_mask = np.sum(np.where(abs(blk.mlp.fc1.weight.data.cpu().numpy())==0., 0., 1.), axis=1)
            blk_dense.mlp.fc1.weight.data = blk.mlp.fc1.weight.data[MLP_mask!=0,:][:,EMB_mask!=0]
            blk_dense.mlp.fc2.weight.data = blk.mlp.fc2.weight.data[EMB_mask!=0,:][:,MLP_mask!=0]
            blk_dense.mlp.fc1.bias.data = blk.mlp.fc1.bias.data[MLP_mask!=0]
            blk_dense.mlp.fc2.bias.data = blk.mlp.fc2.bias.data[EMB_mask!=0]
            
            head_mask = rearrange(blk.attn.qkv.head_mask.weight.data.cpu().numpy(), 'b t -> (b t)')
            #print(head_mask)
            qw = rearrange(blk.attn.qkv.Q.weight.data.cpu().numpy()*head_mask, 'q e h -> (h q) e')
            qb = rearrange(blk.attn.qkv.Q.bias.data.cpu().numpy()*head_mask, 'q h -> (h q)')
            kw = rearrange(blk.attn.qkv.K.weight.data.cpu().numpy()*head_mask, 'q e h -> (h q) e')
            kb = rearrange(blk.attn.qkv.K.bias.data.cpu().numpy()*head_mask, 'q h -> (h q)')
            vw = rearrange(blk.attn.qkv.V.weight.data.cpu().numpy()*head_mask, 'q e h -> (h q) e')
            vb = rearrange(blk.attn.qkv.V.bias.data.cpu().numpy()*head_mask, 'q h -> (h q)')
            pw = rearrange(blk.attn.proj.weight.data.cpu().numpy(), 'e v h -> e (h v)')
            pb = blk.attn.proj.bias.data.cpu().numpy()
    
            QK_mask = np.sum(np.where(abs(qw)==0., 0., 1.), axis=1)
            V_mask = np.sum(np.where(abs(vw)==0., 0., 1.), axis=1)
            #print(blk.attn.scale) 
            #print(np.sum(QK_mask!=0))
            #print(np.sum(V_mask!=0))
            
            blk_dense.attn.qkv.Q.weight.data = torch.from_numpy(qw[QK_mask!=0,:][:,EMB_mask!=0]).float().to(device)
            blk_dense.attn.qkv.K.weight.data = torch.from_numpy(kw[QK_mask!=0,:][:,EMB_mask!=0]).float().to(device)
            blk_dense.attn.qkv.V.weight.data = torch.from_numpy(vw[V_mask!=0,:][:,EMB_mask!=0]).float().to(device)
            blk_dense.attn.proj.proj.weight.data = torch.from_numpy(pw[EMB_mask!=0,:][:,V_mask!=0]).float().to(device)
            blk_dense.attn.qkv.Q.bias.data = torch.from_numpy(qb[QK_mask!=0]).float().to(device)
            blk_dense.attn.qkv.K.bias.data = torch.from_numpy(kb[QK_mask!=0]).float().to(device)
            blk_dense.attn.qkv.V.bias.data = torch.from_numpy(vb[V_mask!=0]).float().to(device)
            blk_dense.attn.proj.proj.bias.data = torch.from_numpy(pb[EMB_mask!=0]).float().to(device)
      
        dense_model.norm.weight.data = student_model.norm.weight.data[EMB_mask!=0]
        dense_model.norm.bias.data = student_model.norm.bias.data[EMB_mask!=0]
        dense_model.head.weight.data = student_model.head.weight.data[:,EMB_mask!=0]
        dense_model.head_dist.weight.data = student_model.head_dist.weight.data[:,EMB_mask!=0]
        dense_model.head.bias.data = student_model.head.bias.data
        dense_model.head_dist.bias.data = student_model.head_dist.bias.data
    
    ################################################
    #exit()
    del student_model
    
    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        print("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer_teacher = create_optimizer(args, teacher_model)
    optimizer = create_optimizer(args, dense_model)

    loss_scaler = None
    if use_amp == 'apex':
        teacher_model, optimizer_teacher = amp.initialize(teacher_model, optimizer_teacher, opt_level='O1')
        dense_model, optimizer = amp.initialize(dense_model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            print('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            print('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            print('AMP not enabled. Training in float32.')
    
    del optimizer_teacher
    
    teacher_model_ema = None
    dense_model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        teacher_model_ema = ModelEma(
            teacher_model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        dense_model_ema = ModelEma(
            dense_model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    teacher_model_without_ddp = teacher_model
    dense_model_without_ddp = dense_model
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            teacher_model = ApexDDP(teacher_model, delay_allreduce=True)
            dense_model = ApexDDP(dense_model, delay_allreduce=True)
        else:
            teacher_model = NativeDDP(teacher_model, device_ids=[args.gpu])  # can use device str in Torch >= 1.1
            dense_model = NativeDDP(dense_model, device_ids=[args.gpu])
        teacher_model_without_ddp = teacher_model.module
        dense_model_without_ddp = dense_model.module

    n_parameters = sum(p.numel() for p in dense_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #utils.print_model_parameters(student_model)
    #exit()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print('nomix2')

    CNN_teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        CNN_teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        CNN_teacher_model.load_state_dict(checkpoint['model'])
        CNN_teacher_model.to(device)
        CNN_teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, CNN_teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    
    orig_criterion = criterion
    distillation_loss = Knowledge_Distillation_Loss(T=args.distillation_tau).cuda()
    criterion = lambda x, y: distillation_loss.get_knowledge_distillation_loss(x, y)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        dense_model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            total_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(dense_model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    
    if args.eval:
        test_stats = evaluate(data_loader_val, dense_model, device)
        if args.asp:
            ASP.prune_trained_model(dense_model, optimizer)
            test_stats = evaluate(data_loader_val, dense_model, device)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'ASP_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': dense_model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': total_epoch,
                        'model_ema': get_state_dict(dense_model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'dim': dim_dict,
                    }, checkpoint_path)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        return
    
    teacher_model.eval()
    max_accuracy = 0.
    
    test_stats = evaluate(data_loader_val, dense_model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': total_epoch,
                 'n_parameters': n_parameters}

    if utils.is_main_process():
        for key in log_stats:
            if 'epoch' not in key:
                writer.add_scalar(key,log_stats[key],total_epoch)
    
    if args.asp:
        total_epoch += 1
        print("Performing Ampere sparsity pruning")
        ASP.prune_trained_model(dense_model, optimizer)
        test_stats = evaluate(data_loader_val, dense_model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
    
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': total_epoch,
                     'n_parameters': n_parameters}
    
        if utils.is_main_process():
            for key in log_stats:
                if 'epoch' not in key:
                    writer.add_scalar(key,log_stats[key],total_epoch)

    main_loss_coeff = args.kl_loss_coeff
    original_loss_coeff = args.original_loss_coeff

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)
        train_stats = finetune_one_epoch(
            args,dense_model, teacher_model, None, criterion, orig_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, dense_model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            main_loss_coeff=main_loss_coeff,original_loss_coeff=original_loss_coeff
        )
        total_epoch += 1
        #lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'ft_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': dense_model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': total_epoch,
                    'model_ema': get_state_dict(dense_model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'dim': dim_dict,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, dense_model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch+1,
                     'n_parameters': n_parameters}

        if utils.is_main_process():
            for key in log_stats:
                if 'epoch' not in key:
                    writer.add_scalar(key,log_stats[key],total_epoch)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
    return
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

