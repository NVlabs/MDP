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
from prune.pruning_fine import *



class NetOptimizer():
    def __init__(self, model, len_train_dataset, train_loader, args, layer_structure=None, gpu=None):
        self.global_step = 0
        self.args = args
        self.gpu = gpu
        self.gpus = args.gpus
        self.is_main = self.gpu is None or self.gpu==0

        self.model = model

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
                                                 {'params': bn_biases, 'lr': 2 * self.lr, 'weight_decay': self.bn_weight_decay}, {'params': bn_params, 'weight_decay': self.bn_weight_decay}],
                                         lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                                         nesterov=self.nesterov)
        
        if not args.disable_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level)
        self.model = DDP(self.model, delay_allreduce=True)
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint)
            model_state_dict = OrderedDict()
            if 'model_state_dict' in checkpoint:
                for k, v in checkpoint['model_state_dict'].items():
                    model_state_dict[k.replace("features_0", "f_0")] = v
            else:
                for k, v in checkpoint.items():
                    model_state_dict[k.replace("features_0", "f_0")] = v
            self.model.load_state_dict(model_state_dict)

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
        self.weight_metric_value = {}

        # schedule the prune number for each epoch
        if args.prune_start:
            self.gs_start = args.prune_start
        else:
            self.gs_start = self.start_epoch

        self.two_to_four = args.two_to_four
        self.two_to_four_spread = args.two_to_four_spread
        self.two_to_four_spread_reverse = args.two_to_four_spread_reverse
        if self.two_to_four:
            self.prune_ratio = 0.5
            self.neuron_norm = None
            self.layer_weight_num = defaultdict(list)
            self.group_mask = {}
            self.prune_target = [0 for _ in range(self.epochs)]
            self.method = args.method
            self.update_metric = (self.method == 2)
        elif self.two_to_four_spread:
            self.prune_ratio = 0.5
            self.neuron_norm = None
            self.layer_weight_num = defaultdict(list)
            self.group_mask = {}
            self.prune_target = [0 for _ in range(self.epochs)]
            self.method = args.method
            self.update_metric = (self.method == 2)
            self.group_count = 0
            self.prune_interval = args.prune_interval
        else:
            if args.prune_end:
                self.gs_end = args.prune_end
            else:
                self.gs_end = self.epochs
            self.prune_interval = args.prune_interval
            self.prune_ratio = args.prune_ratio
            self.prune_target = self.set_prune_target(method=args.prune_method)
            if sum(self.prune_target) == 0:
                self.neuron_norm = defaultdict(list)
                self.layer_weight_num = None
                self.group_mask = None
            else:
                self.neuron_norm = None
                self.layer_weight_num = defaultdict(list)
                self.group_mask = {}

            self.method = args.method
            self.update_metric = (self.method == 2)
            if self.is_main:
                print('self.prune_target: ', self.prune_target)


    def set_prune_target(self, method='exp'):
        """
            schedule the target number of weights to prune for each epoch
        Returns:
            to_prune: List with size of total epochs.
                      Each item is the number of weights to prune at the corresponding epoch.
        """
        if self.prune_ratio == 0.:
            return [0 for _ in range(self.epochs)]
        total_conv_num = count_all_weights(self.layers)
        total_prune_num = int(total_conv_num * self.prune_ratio)
        if self.is_main:
            print("Total weights: {}, total prune target weights: {}".format(total_conv_num, total_prune_num))
        
        if method == 'exp':
            to_prune = [math.exp(x/5.0) for x in range(0, self.gs_end-self.gs_start)]
            scale = total_prune_num / sum(to_prune)
            to_prune = [int(x*scale) for x in to_prune][::-1]
        elif method == 'exp2':  # exp schedule proposed in FORCE
            final_remain = total_conv_num - total_prune_num
            kt = [0 for _ in range(self.gs_end - self.gs_start + 1)]
            T = self.gs_end - self.gs_start
            for t in range(0, self.gs_end - self.gs_start + 1):
                alpha = t / T
                kt[t] = math.exp(alpha * math.log(final_remain) + (1 - alpha) * math.log(total_conv_num))
            to_prune = [int(kt[t] - kt[t + 1]) for t in range(0, self.gs_end - self.gs_start)]
        elif method == 'linear':
            to_prune = list(range(1, self.gs_end-self.gs_start+1))
            scale = total_prune_num / sum(to_prune)
            to_prune = [int(x*scale) for x in to_prune][::-1]
        elif method == 'uniform':
            to_prune = [1 for _ in range(0, self.gs_end-self.gs_start)]
            scale = total_prune_num / sum(to_prune)
            to_prune = [int(x*scale) for x in to_prune][::-1]
        remain = total_prune_num - sum(to_prune)
        for i in range(remain):
            to_prune[i % len(to_prune)] += remain//abs(remain)
        assert sum(to_prune) == total_prune_num

        to_prune_interval = []
        for prune_epoch in to_prune:
            to_prune_interval.extend([prune_epoch] + [0]*(self.prune_interval - 1))
        to_prune = to_prune_interval
        self.gs_end = self.gs_start + len(to_prune)
        to_prune = [0 for _ in range(self.gs_start)] + to_prune + [0 for _ in range(self.gs_end, self.epochs)]
        return to_prune


    def optimize_net(self):
        """
            Optimize the whole neural network
        """
        #metric_to_save = defaultdict(list)
                    
        pre_pruned_num = 0
        pre_target_num = 0

        for epoch_id in tqdm(range(0, 10)):
            if self.lr_at[epoch_id] != 0:
                adjust_learning_rate2(self.optimizer, 0.0001)

            # train epoch
            loss = self.train_epoch(epoch_id)
            
            self.writer.add_scalar('train/loss', loss, epoch_id)
            self.writer.add_scalar('train/LR', self.optimizer.param_groups[1]['lr'], epoch_id)
            
            # Record the weight count before pruning.
            ori_conv_weights = count_non_zero_weights(self.layers, self.layer_weight_num)
            self.writer.add_scalar('weight_num', ori_conv_weights, epoch_id)
            # for name, layer_metric in self.weight_metric_value.items():
            #     metric_to_save[name].append(layer_metric.cpu().numpy().squeeze())

            # apply regularization
            # One-shot two-to-four pruning
            if self.two_to_four and epoch_id == self.gs_start:
                # Checkpoint before pruning
                if self.is_main:
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_before_prune.pth'.format(epoch_id))
                    save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
                    print('Saved model to {}.'.format(save_path))

                # Apply 2:4 pruning
                total_pruned_num = apply_two_to_four_pruning(self.layers, self.weight_metric_value, self.group_mask, self.is_main)
                cur_conv_weights = count_non_zero_weights(self.layers)
                if self.is_main:
                    print(ori_conv_weights, total_pruned_num, cur_conv_weights)

                # Checkpoint after pruning
                if self.is_main:
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_after_prune.pth'.format(epoch_id))
                    save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
                    print('Saved model to {}.'.format(save_path))
            
            elif self.two_to_four_spread and epoch_id >= self.gs_start and self.group_count < len(self.groups):
                if (epoch_id - self.gs_start) % self.prune_interval == 0:
                    if self.two_to_four_spread_reverse:
                        curr_count = len(self.groups) - 1 - self.group_count
                    else:
                        curr_count = self.group_count
                    total_pruned_num = apply_two_to_four_pruning_one_by_one(self.layers, self.groups[curr_count], self.weight_metric_value, self.group_mask, self.is_main)
                    cur_conv_weights = count_non_zero_weights(self.layers)
                    self.group_count += 1
                    if self.is_main:
                        print(ori_conv_weights, total_pruned_num, cur_conv_weights)

                    # Checkpoint after pruning
                    if self.is_main:
                        save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_after_prune.pth'.format(epoch_id))
                        save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
                        print('Saved model to {}.'.format(save_path))

            elif self.prune_target[epoch_id] > 0:
                # Apply actual pruning
                total_pruned_num = apply_pruning(self.layers, self.weight_metric_value,
                                                 self.prune_target[epoch_id],
                                                 pre_pruned_num, pre_target_num,
                                                 self.group_mask, self.is_main)
                pre_pruned_num += total_pruned_num
                pre_target_num += self.prune_target[epoch_id]
                cur_conv_weights = count_non_zero_weights(self.layers)
                if self.is_main:
                    print('Total actually pruned number {} vs. Scheduled number {}'.format(pre_pruned_num, pre_target_num))
                    print(ori_conv_weights, total_pruned_num, cur_conv_weights)

                # Checkpoint after pruning
                if self.is_main:
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_after_prune.pth'.format(epoch_id))
                    save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
                    print('Saved model to {}.'.format(save_path))
            
            if self.is_main and epoch_id in self.checkpoint_epochs:
                save_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch_id))
                save_checkpoint(epoch_id, self.model, self.optimizer, save_path)
                print('Saved model to {}.'.format(save_path))

        for epoch_id in tqdm(range(self.start_epoch, self.epochs)):            
            if self.lr_at[epoch_id] != 0:
                adjust_learning_rate2(self.optimizer, self.lr_at[epoch_id])


            # train epoch
            loss = self.train_epoch(epoch_id)
            
            self.writer.add_scalar('train/loss', loss, epoch_id+10)
            self.writer.add_scalar('train/LR', self.optimizer.param_groups[1]['lr'], epoch_id+10)
            
            # Record the weight count before pruning.
            ori_conv_weights = count_non_zero_weights(self.layers, self.layer_weight_num)
            self.writer.add_scalar('weight_num', ori_conv_weights, epoch_id+10)

            # apply regularization
            # One-shot two-to-four pruning
            if self.two_to_four and epoch_id == self.gs_start:
                # Checkpoint before pruning
                if self.is_main:
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_before_prune.pth'.format(epoch_id+10))
                    save_checkpoint(epoch_id+10, self.model, self.optimizer, save_path)
                    print('Saved model to {}.'.format(save_path))

                # Apply 2:4 pruning
                total_pruned_num = apply_two_to_four_pruning(self.layers, self.weight_metric_value, self.group_mask, self.is_main)
                cur_conv_weights = count_non_zero_weights(self.layers)
                if self.is_main:
                    print(ori_conv_weights, total_pruned_num, cur_conv_weights)

                # Checkpoint after pruning
                if self.is_main:
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_after_prune.pth'.format(epoch_id+10))
                    save_checkpoint(epoch_id+10, self.model, self.optimizer, save_path)
                    print('Saved model to {}.'.format(save_path))
            
            elif self.two_to_four_spread and epoch_id >= self.gs_start and self.group_count < len(self.groups):
                if (epoch_id - self.gs_start) % self.prune_interval == 0:
                    if self.two_to_four_spread_reverse:
                        curr_count = len(self.groups) - 1 - self.group_count
                    else:
                        curr_count = self.group_count
                    total_pruned_num = apply_two_to_four_pruning_one_by_one(self.layers, self.groups[curr_count], self.weight_metric_value, self.group_mask, self.is_main)
                    cur_conv_weights = count_non_zero_weights(self.layers)
                    self.group_count += 1
                    if self.is_main:
                        print(ori_conv_weights, total_pruned_num, cur_conv_weights)

                    # Checkpoint after pruning
                    if self.is_main:
                        save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_after_prune.pth'.format(epoch_id+10))
                        save_checkpoint(epoch_id+10, self.model, self.optimizer, save_path)
                        print('Saved model to {}.'.format(save_path))

            elif self.prune_target[epoch_id] > 0:
                # Apply actual pruning
                total_pruned_num = apply_pruning(self.layers, self.weight_metric_value,
                                                 self.prune_target[epoch_id],
                                                 pre_pruned_num, pre_target_num,
                                                 self.group_mask, self.is_main)
                pre_pruned_num += total_pruned_num
                pre_target_num += self.prune_target[epoch_id]
                cur_conv_weights = count_non_zero_weights(self.layers)
                if self.is_main:
                    print('Total actually pruned number {} vs. Scheduled number {}'.format(pre_pruned_num, pre_target_num))
                    print(ori_conv_weights, total_pruned_num, cur_conv_weights)

                # Checkpoint after pruning
                if self.is_main:
                    save_path = os.path.join(self.checkpoint_dir, 'epoch_{}_after_prune.pth'.format(epoch_id+10))
                    save_checkpoint(epoch_id+10, self.model, self.optimizer, save_path)
                    print('Saved model to {}.'.format(save_path))
            
            if self.is_main and epoch_id in self.checkpoint_epochs:
                save_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch_id+10))
                save_checkpoint(epoch_id+10, self.model, self.optimizer, save_path)
                print('Saved model to {}.'.format(save_path))
                

        if self.is_main:
            # analyze the pruned model
            self.final_model_analysis()
            if self.neuron_norm is not None:
                with open('{}/layer_avg_norm.pkl'.format(self.output_dir), 'wb') as outfile:
                    pkl.dump(self.neuron_norm, outfile)
            if self.layer_weight_num is not None:
                with open('{}/layer_weight_num.pkl'.format(self.output_dir), 'wb') as outfile:
                    pkl.dump(self.layer_weight_num, outfile)
            # with open('{}/group_metric.pkl'.format(self.output_dir), 'wb') as outfile:
            #     pkl.dump(metric_to_save, outfile)
            if self.group_mask is not None:
                with open('{}/group_mask.pkl'.format(self.output_dir), 'wb') as outfile:
                    pkl.dump(self.group_mask, outfile)
            save_path = os.path.join(self.output_dir, 'net_final.pth')
            save_checkpoint(self.epochs, self.model, self.optimizer, save_path)
            print('Saved model to {}.'.format(save_path))


    def train_epoch(self, epoch_id):
        """
            Optimize the neural network in one epoch
        Return:
            averaged loss of this epoch
        """
        losses = AverageMeter()
        self.weight_metric_value = {}

        for batch_id, (images, boxes, labels, _) in enumerate(self.train_loader):
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

            # Backward prop.
            if not self.args.disable_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Clip gradients, if necessary
            if self.grad_clip is not None:
                clip_gradient(optimizer, args.grad_clip)


            # mask gradients and gradient momentum buffer
            if self.group_mask is not None:
                for group, mask in self.group_mask.items():
                    weight = self.layers[group].weight
                    weight.grad.data.mul_(mask)
                    param_state = self.optimizer.state[weight]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].data.mul_(mask)


            # update the importance
            if self.update_metric:
                update_weight_metric(self.layers, self.groups, layer_bn=self.layer_bn, layer_gate=self.layer_gate, method=self.method, weight_metric=self.weight_metric_value)
                
            # Update model
            self.optimizer.step()

            #self.writer.add_scalar('train_iter/loss', loss.item(), self.iters)
            self.iters += 1

            losses.update(loss.item(), images.size(0))

        del predicted_locs, predicted_scores, images, boxes, labels

        if self.update_metric:
            # average the importance over the entire epoch
            for group_name in self.weight_metric_value.keys():
                self.weight_metric_value[group_name] = torch.div(self.weight_metric_value[group_name], len(self.train_loader))
        else:
            update_weight_metric(self.layers, self.groups, layer_bn=self.layer_bn, layer_gate=self.layer_gate, method=self.method, weight_metric=self.weight_metric_value)

        return losses.avg


    def final_model_analysis(self):
        """
            Analyze the pruned network
        """
        total_conv_weights = 0
        for name, layer in self.layers.items():
            if is_conv(layer) and (not 'predictor' in name):
                conv_weights = count_non_zero(layer.weight.data.view(-1))
                print('{} has {} weights.'.format(name, conv_weights))
                total_conv_weights += conv_weights
        print('The final network has {} weights in total.'.format(total_conv_weights))
ame, conv_weights))
                total_conv_weights += conv_weights
        print('The final network has {} weights in total.'.format(total_conv_weights))
