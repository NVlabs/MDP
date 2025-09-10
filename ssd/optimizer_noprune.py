import numpy as np
import torch
import torch.nn as nn
import time
import os
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import math
from losses import MultiBoxLoss
from tqdm import tqdm
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import torch.distributed as dist
import pytz
from datetime import datetime
import pickle as pkl
from collections import OrderedDict

from utils.utils import *
from prune.latency_targeted_pruning import apply_latency_target_pruning, apply_latency_target_regrowing
from prune.pruning import *
from masking import *
from copy import deepcopy

PRUNE_INVERVAL = 50

def set_masks(layers, group_mask, layer_bn=None):
    """Zero the neuron weights according to the mask."""
    if group_mask is None:
        return
    for group_name, mask in group_mask.items():
        for layer_name in group_name:
            layer = layers[layer_name]
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                # Mask conv neurons
                if not isinstance(layer, nn.ConvTranspose2d):
                    set_mask(layer, 'weight', mask.view(-1, 1, 1, 1))
                else:
                    set_mask(layer, 'weight', mask.view(1, -1, 1, 1))
                if layer.bias is not None:
                    set_mask(layer, 'bias', mask)
                if layer_bn is not None:
                    bn_layer = layers[layer_bn[layer_name]]
                    set_mask(bn_layer, 'weight', mask)
                    set_mask(bn_layer, 'bias', mask)



# In the freeze mask, a value of 1 means setting the gradients to 0
def freeze_gradients(layers, freeze_mask, optimizer, layer_bn=None):
    # print(freeze_gradients)

    if freeze_mask is None:
        return
    for group_name, mask in freeze_mask.items():
        mask = 1 - mask
        for layer_name in group_name:
            layer = layers[layer_name]
            layer.weight_orig.grad.data.mul_(mask.view(-1, 1, 1, 1))
            param_state = optimizer.state[layer.weight_orig]
            if 'momentum_buffer' in param_state:
                param_state['momentum_buffer'].data.mul_(mask.view(-1, 1, 1, 1))

            if hasattr(layer, "bias") and layer.bias is not None:
                print("Has Bias.")
            if hasattr(layer, "bias_orig") and layer.bias_orig is not None:
                layer.bias_orig.grad.data.mul_(mask.view(-1, 1, 1, 1))
                param_state = optimizer.state[layer.bias_orig]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].data.mul_(mask)

            if layer_bn is not None:
                bn_layer = layers[layer_bn[layer_name]]
                bn_layer.weight_orig.grad.data.mul_(mask)
                bn_layer.bias_orig.grad.data.mul_(mask)

                param_state = optimizer.state[bn_layer.weight_orig]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].data.mul_(mask)
                param_state = optimizer.state[bn_layer.bias_orig]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].data.mul_(mask)


# Masking layer_bn
def add_masking_hooks(layers, layer_bn):
    for layer_name, layer in layers.items():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            add_attr_masking(layer, 'weight', ParameterMaskingType.Hard, True)
            if hasattr(layer, "bias") and layer.bias is not None:
                add_attr_masking(layer, 'bias', ParameterMaskingType.Hard, True)
            if layer_bn is not None and layer_name in layer_bn:
                bn_layer = layers[layer_bn[layer_name]]
                add_attr_masking(bn_layer, 'weight', ParameterMaskingType.Hard, True)
                add_attr_masking(bn_layer, 'bias', ParameterMaskingType.Hard, True)


def remove_masking_hooks(layers, layer_bn):
    for layer_name, layer in layers.items():
        if hasattr(layer, 'weight_orig'):
            remove_attr_masking(layer, 'weight')
        if hasattr(layer, 'bias_orig'):
            remove_attr_masking(layer, 'bias')
        if layer_bn is not None and layer_name in layer_bn:
            bn_layer = layers[layer_bn[layer_name]]
            remove_attr_masking(bn_layer, 'weight')
            remove_attr_masking(bn_layer, 'bias')



class NetOptimizer():
    def __init__(self, model, len_train_dataset, train_loader, args, layer_structure=None, gpu=None):
        self.global_step = 0
        self.args = args
        self.regrow_ratio = args.regrow_ratio
        self.gpu = gpu
        self.gpus = args.gpus
        self.is_main = self.gpu is None or self.gpu==0

        self.model = model
        self.prune_per_epoch = args.prune_per_epoch

        self.train_loader = train_loader
        self.batch_size = args.batch_size

        self.lr = args.lr
        self.lrs = args.lrs
        self.lr_schedule = args.lr_schedule
        self.decay_rates = args.decay_rates
        self.grad_clip = args.grad_clip

        self.weight_decay = args.weight_decay
        self.bn_weight_decay = args.bn_weight_decay
        self.nesterov = args.nesterov
        self.momentum = args.momentum
        self.epochs = args.epochs

        if not args.disable_linear_lr:
            self.lr_at = []
            curr_epoch = 0
            curr_lr = self.lr
            for epoch, new_lr in zip(self.lr_schedule, self.lrs[1:]):
                self.lr_at.extend(list(np.linspace(curr_lr, new_lr, epoch-curr_epoch, endpoint=False)))
                curr_lr, curr_epoch = new_lr, epoch
            self.lr_at.append(self.lrs[-1])
        else:
            self.lr_at = [0] * self.epochs
            for epoch, new_lr in zip(self.lr_schedule, self.lrs[1:]):
                self.lr_at[epoch] = new_lr

        self.checkpoint_dir = args.checkpoint_dir
        self.disable_gating = not args.enable_gating

        self.checkpoint_epochs = [int(self.epochs*y) for y in [0, 0.5, 0.6, 0.7, 0.8, 0.9]] + list(range(self.epochs-30, self.epochs))
        if args.checkpoint_all:
            self.checkpoint_epochs = list(range(self.epochs))

        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        bn_params = list()
        bn_biases = list()
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if 'bn' in param_name:
                    if param_name.endswith('.bias'):
                        bn_biases.append(param)
                    else:
                        bn_params.append(param)
                elif param_name.endswith('.bias'):
                    biases.append(param)
                elif "gate" not in param_name:
                    not_biases.append(param)

        self.optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * self.lr}, {'params': not_biases}, \
                                                 {'params': bn_biases, 'lr': 2 * self.lr,
                                                  'weight_decay': self.bn_weight_decay},
                                                 {'params': bn_params, 'weight_decay': self.bn_weight_decay}],
                                         lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                                         nesterov=self.nesterov)

        if not args.disable_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level)
        self.model = DDP(self.model, delay_allreduce=True)

        self.layers = extract_layers(self.model, get_conv=True, get_bn=True, get_gate=not self.disable_gating)

        self.criterion = MultiBoxLoss(priors_cxcy=self.model.module.priors_cxcy).cuda()

        self.start_epoch = 0
        self.iters = 0
        if self.gpu:
            self.writer = SummaryWriter(os.path.join(args.tensorboard_dir, str(self.gpu)))
        else:
            self.writer = SummaryWriter(os.path.join(args.tensorboard_dir, '0'))


        self.output_dir = args.output_dir

        self.layer_bn, self.layer_gate, self.groups = layer_structure
        if self.disable_gating:
            self.layer_gate = None

        # save the info for further analysis
        self.neuron_metric_value = {}

        # schedule the prune number for each epoch
        self.prune_ratio = args.prune_ratio
        if args.prune_start:
            self.gs_start = args.prune_start
        else:
            self.gs_start = self.start_epoch
        if args.prune_end:
            self.gs_end = args.prune_end
        else:
            self.gs_end = self.epochs
        self.prune_interval = args.prune_interval

        if not args.no_prune:
            total_conv_num = count_non_zero_neurons(self.layers)
            assert args.prune_method in ['exp', 'exp2', 'linear']
            self.prune_target = self.set_prune_target(total_conv_num, args.prune_ratio,
                                             PRUNE_INVERVAL,
                                             args.prune_method)
            # self.prune_target = self.set_prune_target(total_conv_num, args.prune_ratio,
            #                                  self.gs_start, self.gs_end,
            #                                  args.prune_method)

        if self.prune_target is None:
            self.neuron_norm = defaultdict(list)
            self.neuron_norm = None
            self.layer_neuron_num = None
            self.group_mask = None
        else:
            self.neuron_norm = None
            self.layer_neuron_num = defaultdict(list)
            self.group_mask = {}

        self.method = args.method
        self.update_metric = self.method not in [0, 1, 6]

        self.lookup_table, self.fmap_table = None, None
        self.pre_group, self.aft_group_list = None, None
        if self.method == 9 or self.method == 10 or self.method >= 20:  # latency aware pruning / latency targeted pruning
            from prune.prune_config_with_structure import PruneConfigReader
            config_reader = PruneConfigReader()
            config_reader.set_prune_setting(args.reg_conf.replace('.json', '_structure.json'))
            _, _, _, self.pre_group, self.aft_group_list = config_reader.get_layer_structure()
            latency_file_path = '/home/cudnn_v7.4_conv_LUT_repeat_100_step_2_scope_2048_batch64_forward_ssd512_resnet50.pkl'
            with open(latency_file_path, 'rb') as f:
                self.lookup_table = pkl.load(f)
            import json
            if not args.backbone_only:
                with open('{}_fmap.json'.format(args.arch), 'r') as f:
                    self.fmap_table = json.load(f)
            else:
                with open('{}_backbone_fmap.json'.format(args.arch), 'r') as f:
                    self.fmap_table = json.load(f)
            from prune.latency_targeted_pruning import load_group_size
            load_group_size(args.arch, args.backbone_only)

        if self.method >= 20:
            from prune.latency_targeted_pruning import set_latency_prune_target, get_total_latency

            # chunk of code if you want to figure out latency of pruned model
            # group_mask_path = '/workspace/marvin-latency/ssd512_resnet50_prune0.449_backbone_0-200/2021-05-02_19-37-04/group_mask.pkl'
            # with open(group_mask_path, 'rb') as f:
            #     self.group_mask = pkl.load(f)
            # prune_latency, backbone_latency, head_latency = get_total_latency(self.layers, self.groups, self.pre_group, self.group_mask, self.fmap_table, self.lookup_table)
            # print(prune_latency, "prune latency of model!")
            # self.group_mask = {}
            
            initial_latency, backbone_latency, head_latency = get_total_latency(self.layers, self.groups, self.pre_group, self.group_mask, self.fmap_table, self.lookup_table)
            self.initial_latency = initial_latency
            # if args.prune_per_epoch:
            self.prune_target = set_latency_prune_target(initial_latency, PRUNE_INVERVAL, args.target_latency, args.prune_ratio, mode=args.prune_method)
            print("Latency Target")
            print(self.prune_target)
    # def set_prune_target(self, total_conv_num, prune_ratio, gs_start, gs_end, mode='exp'):
    #     """
    #         schedule the target number of neurons to prune for each epoch
    #     Returns:
    #         to_prune: List with size of total epochs.
    #                   Each item is the number of neurons to prune at the corresponding epoch.
    #     """
    #     total_prune_num = int(total_conv_num * prune_ratio)
    #     if mode == 'exp':
    #         to_prune = [math.exp(x/20.0) for x in range(0, gs_end-gs_start)]
    #         scale = total_prune_num / sum(to_prune)
    #         to_prune = [int(x*scale) for x in to_prune][::-1]
    #     elif mode == 'exp2':  # exp schedule proposed in FORCE
    #         final_remain = total_conv_num - total_prune_num
    #         kt = [0 for _ in range(gs_end - gs_start + 1)]
    #         T = gs_end - gs_start
    #         for t in range(0, gs_end - gs_start + 1):
    #             alpha = t / T
    #             kt[t] = math.exp(alpha * math.log(final_remain) + (1 - alpha) * math.log(total_conv_num))
    #         to_prune = [int(kt[t] - kt[t + 1]) for t in range(0, gs_end - gs_start)]
    #     else:  # linear mode
    #         to_prune = [total_prune_num//(gs_end-gs_start) for _ in range(0, gs_end-gs_start)]
    #     remain = total_prune_num - sum(to_prune)
    #     for i in range(remain):
    #         to_prune[i] += remain//abs(remain)
    #     assert sum(to_prune) == total_prune_num
    #     return to_prune

    def set_prune_target(self, total_conv_num, prune_ratio, prune_inverval, mode='exp'):
        """
            schedule the target number of neurons to prune for each epoch
        Returns:
            to_prune: List with size of total epochs.
                      Each item is the number of neurons to prune at the corresponding epoch.
        """
        total_prune_num = int(total_conv_num * prune_ratio)
        if mode == 'exp':
            to_prune = [math.exp(x/20.0) for x in range(0, prune_inverval)]
            scale = total_prune_num / sum(to_prune)
            to_prune = [int(x*scale) for x in to_prune][::-1]
        elif mode == 'exp2':  # exp schedule proposed in FORCE
            final_remain = total_conv_num - total_prune_num
            kt = [0 for _ in range(prune_inverval + 1)]
            T = prune_inverval
            for t in range(0, prune_inverval + 1):
                alpha = t / T
                kt[t] = math.exp(alpha * math.log(final_remain) + (1 - alpha) * math.log(total_conv_num))
            to_prune = [int(kt[t] - kt[t + 1]) for t in range(0, prune_inverval)]
        else:  # linear mode
            to_prune = [total_prune_num//(prune_inverval) for _ in range(0, prune_inverval)]
        remain = total_prune_num - sum(to_prune)
        for i in range(remain):
            to_prune[i] += remain//abs(remain)
        assert sum(to_prune) == total_prune_num
        return to_prune

    def optimize_net(self):
        """
            Optimize the whole neural network
        """
        print("No Pruning.")
        print("Just Train the Baseline.")
        pre_pruned_num = 0
        pre_target_num = 0
        self.neuron_metric_value = {}
        self.regrown_neuron_metric_value = {}
        self.freeze = False
        self.prev_latency = self.initial_latency
        # add_masking_hooks(self.layers, self.layer_bn)
        for epoch_id in tqdm(range(0, self.epochs)):            
            if self.lr_at[epoch_id] != 0:
                adjust_learning_rate2(self.optimizer, self.lr_at[epoch_id])

            if epoch_id == self.gs_start:
                self.to_grow = False
            # train epoch
            loss = self.train_epoch(epoch_id)
            
            self.writer.add_scalar('train/loss', loss, epoch_id)
            self.writer.add_scalar('train/LR', self.optimizer.param_groups[1]['lr'], epoch_id)
            
            # Record the neuron count before pruning.
            # ori_conv_neurons = count_non_zero_neurons(self.layers, self.layer_neuron_num, self.neuron_norm)
            # self.writer.add_scalar('neuron_num', ori_conv_neurons, epoch_id)

            #prune at the end of every epoch  
            # if self.prune_target is not None and self.gs_start <= epoch_id < self.gs_end and self.prune_target[epoch_id-self.gs_start] > 0:
            # if self.prune_target is not None and len(self.prune_target) > 0 and self.gs_start <= epoch_id and not self.to_grow:
            # # if self.prune_target is not None and len(self.prune_target) > 0 and self.gs_start <= epoch_id and self.prune_target[0] > 0 and not self.to_grow:
            #     target_latency = self.prune_target.pop(0)
                
            #     total_pruned_num, total_latency = apply_latency_target_pruning(self.layers, self.layer_bn, self.layer_gate, self.neuron_metric_value, 
            #                                                     target_latency, self.group_mask,
            #                                                     self.groups, self.pre_group, self.aft_group_list, self.fmap_table, self.lookup_table, self.method)
            #     set_masks(self.layers, self.group_mask, self.layer_bn)
            #     # set_masks(layers, group_mask, layer_bn, mask_bias=True)
            #     total_latency = total_latency[0]
            #     print(f"Latency after pruning at global step {epoch_id}: {total_latency}.")
            #     print(f"Previous Latency: {self.prev_latency}")
            #     self.latency_change = abs(self.prev_latency - total_latency)
            #     print(f"Latency Change: {self.latency_change}")
            #     self.prev_latency = total_latency

            #     for k, v in self.neuron_metric_value.items():
            #         self.neuron_metric_value[k][:] = 0
            #     for k, v in self.regrown_neuron_metric_value.items():
            #         self.regrown_neuron_metric_value[k][:] = 0

            #     self.to_grow = True
            #     self.freeze = True
            #     if len(self.prune_target) == 0:
            #         self.freeze = False
            # elif self.prune_target is not None and len(self.prune_target) > 0 and self.gs_start <= epoch_id and self.to_grow:
            #     self.latency_budget = self.latency_change * self.regrow_ratio
            #     print(f"Available latency budget for regrowing at global step {epoch_id}: {self.latency_budget}.")
            #     total_pruned_num, total_latency, regrow_mask = apply_latency_target_regrowing(self.layers, self.layer_bn, self.layer_gate, self.regrown_neuron_metric_value, 
            #                                                     self.latency_budget, self.group_mask,
            #                                                     self.groups, self.pre_group, self.aft_group_list, self.fmap_table, self.lookup_table, self.method)
            #     set_masks(self.layers, self.group_mask, self.layer_bn)
            #     total_latency = total_latency[0]
            #     # set_masks(layers, group_mask, layer_bn, mask_bias=True)
            #     print(f"Latency after regrowing at global step {epoch_id}: {total_latency}.")
            #     for k, v in self.neuron_metric_value.items():
            #         self.neuron_metric_value[k][:] = 0
            #     for k, v in self.regrown_neuron_metric_value.items():
            #         self.regrown_neuron_metric_value[k][:] = 0
            #     self.to_grow = False
            #     self.freeze = False

            if self.is_main and epoch_id in self.checkpoint_epochs:
                save_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch_id))
                save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
                print('Saved model to {}.'.format(save_path))

        # if self.is_main:
        #     # analyze the pruned model
        #     # self.final_model_analysis()
        #     if self.neuron_norm is not None:
        #         with open('{}/layer_avg_norm.pkl'.format(self.output_dir), 'wb') as outfile:
        #             pkl.dump(self.neuron_norm, outfile)
        #     if self.layer_neuron_num is not None:
        #         with open('{}/layer_neuron_num.pkl'.format(self.output_dir), 'wb') as outfile:
        #             pkl.dump(self.layer_neuron_num, outfile)
        #     if self.group_mask is not None:
        #         with open('{}/group_mask.pkl'.format(self.output_dir), 'wb') as outfile:
        #             pkl.dump(self.group_mask, outfile)
        #     save_path = os.path.join(self.output_dir, 'net_final.pth')
        #     torch.save(self.model.state_dict(), save_path)
        #     print('Saved model to {}.'.format(save_path))
        if self.group_mask is not None:
            with open('{}/group_mask.pkl'.format(self.output_dir), 'wb') as outfile:
                pkl.dump(self.group_mask, outfile)
        save_path = os.path.join(self.checkpoint_dir, 'net_final_unclean.pth')
        save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
        print('Saved model to {}.'.format(save_path))
        # remove_masking_hooks(self.layers, self.layer_bn)
        # save_path = os.path.join(self.output_dir, 'net_final_clean.pth')
        # torch.save(self.model.state_dict(), save_path)
        # print('Saved model to {}.'.format(save_path))
        

    def train_epoch(self, epoch_id):
        """
            Optimize the neural network in one epoch
        Return:
            averaged loss of this epoch
        """
        losses = AverageMeter()
        freeze_mask = None
        for batch_id, (images, boxes, labels, _) in enumerate(self.train_loader):
            # Temporarily reactivate
            # if batch_id == len(self.train_loader) // 2 and self.freeze:
            #     print("Temporarily Activating and Start Freezing.")
            # # if batch_id == 10 and self.freeze:
            #     for k, v in self.neuron_metric_value.items():
            #         self.neuron_metric_value[k][:] = 0
            #     for k, v in self.regrown_neuron_metric_value.items():
            #         self.regrown_neuron_metric_value[k][:] = 0
            #     ones_mask = deepcopy(self.group_mask)
            #     for group_name in ones_mask.keys():
            #         ones_mask[group_name][:] = 1
            #     set_masks(self.layers, ones_mask, self.layer_bn)
            #     freeze_mask = self.group_mask
            
            self.model.train()
            # Move to default device
            images = images.cuda(non_blocking=True)  # (batch_size (N), 3, 300, 300)
            boxes = [b.cuda(non_blocking=True) for b in boxes]
            labels = [l.cuda(non_blocking=True) for l in labels]

            #self.optimizer.zero_grad()
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            # Forward prop.
            predicted_locs, predicted_scores = self.model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = self.criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
            if torch.isnan(loss) or loss.grad_fn is None:
                if loss.grad_fn is None:
                    print("loss grad fn is none")
                print("NAN in loss, skipping the batch and update")
                #reduced_loss, prec1, prec5 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                continue
            # Backward prop.
            if not self.args.disable_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # Clip gradients, if necessary
            if self.grad_clip is not None:
                clip_gradient(self.optimizer, args.grad_clip)

            has_nan = False
            for w in self.model.parameters():
                if torch.isnan(w.grad.data.mean()):
                    has_nan = True
                    break
            if has_nan:
                print("NAN in grad, skipping the batch and update")
                #reduced_loss, prec1, prec5 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                continue


            # if self.group_mask is not None:
            #     for group, mask in self.group_mask.items():
            #         for layer_name in group:
            #             weight = self.layers[layer_name].weight
            #             assert weight.grad.data[mask==0.].sum().item() == 0.
            #             weight.grad.data.mul_(mask.view(-1, 1, 1, 1))
            #             param_state = self.optimizer.state[weight]
            #             if 'momentum_buffer' in param_state:
            #                 param_state['momentum_buffer'].data.mul_(mask.view(-1, 1, 1, 1))
            #             bias = self.layers[layer_name].bias
            #             if bias is not None:
            #                 bias.grad.data.mul_(mask)
            #                 param_state = self.optimizer.state[bias]
            #                 if 'momentum_buffer' in param_state:
            #                     param_state['momentum_buffer'].data.mul_(mask)
            #             if self.layer_bn is not None:
            #                 self.layers[self.layer_bn[layer_name]].weight.grad.data.mul_(mask)
            #                 self.layers[self.layer_bn[layer_name]].bias.grad.data.mul_(mask)
            #                 param_state = self.optimizer.state[self.layers[self.layer_bn[layer_name]].weight]
            #                 if 'momentum_buffer' in param_state:
            #                     param_state['momentum_buffer'].data.mul_(mask)
            #                 param_state = self.optimizer.state[self.layers[self.layer_bn[layer_name]].bias]
            #                 if 'momentum_buffer' in param_state:
            #                     param_state['momentum_buffer'].data.mul_(mask)


            # if freeze_mask is not None:
            #     freeze_gradients(self.layers, freeze_mask, self.optimizer, self.layer_bn)

            # update the importance
            # if self.update_metric:
            #     update_neuron_metric(self.layers, self.groups, layer_bn=self.layer_bn, layer_gate=self.layer_gate, method=self.method, neuron_metric=self.neuron_metric_value)
            #     update_neuron_metric(self.layers, self.groups, layer_bn=self.layer_bn, layer_gate=self.layer_gate, method=self.method, neuron_metric=self.regrown_neuron_metric_value)

            # Update model
            self.optimizer.step()

            #self.writer.add_scalar('train_iter/loss', loss.item(), self.iters)
            self.iters += 1

            losses.update(loss.item(), images.size(0))

            #if epoch_id == 1:

        del predicted_locs, predicted_scores, images, boxes, labels

        if self.update_metric:
            # average the importance over the entire epoch
            for group_name in self.neuron_metric_value.keys():
                self.neuron_metric_value[group_name] = torch.div(self.neuron_metric_value[group_name], len(self.train_loader))
            for group_name in self.regrown_neuron_metric_value.keys():
                self.regrown_neuron_metric_value[group_name] = torch.div(self.regrown_neuron_metric_value[group_name], len(self.train_loader))
        else:
            update_neuron_metric(self.layers, self.groups, layer_bn=self.layer_bn, layer_gate=self.layer_gate, method=self.method, neuron_metric=self.neuron_metric_value)

        return losses.avg


    def final_model_analysis(self):
        """
            Analyze the pruned network
        """
        total_conv_neuron = 0
        for name, layer in self.layers.items():
            if is_conv(layer) and (not 'predictor' in name):
                norm = layer.weight.data.view(layer.weight.data.size(0), -1).norm(dim=1)
                conv_neuron = count_non_zero(norm)
                print('{} has {} neurons.'.format(name, conv_neuron))
                total_conv_neuron += conv_neuron
        print('The final network has {} neurons in total.'.format(total_conv_neuron))
