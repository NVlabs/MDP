
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.models as models_torch
from collections import defaultdict
import pickle as pkl

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from utils.utils import AverageMeter, count_non_zero_neurons, accuracy, reduce_tensor
from prune.pruning import apply_pruning, update_neuron_metric, hybrid_norm_importance, update_neuron_metric_mdp
from prune.latency_targeted_pruning_mdp import apply_latency_target_pruning
from prune.mdp_func import *

# PRUNE_STEPS = 30
PRUNE_STEPS = 1
ACCUM = int(1200 // PRUNE_STEPS)


def get_optimizer(parameters, fp16, lr, momentum, weight_decay,
                  nesterov=False,
                  static_loss_scale=1., dynamic_loss_scale=False,
                  bn_weight_decay=False):

    print(" ! Weight decay applied to BN parameters: {}".format(bn_weight_decay))
    bn_params = [v for n, v in parameters if 'bn' in n]
    rest_params = [v for n, v in parameters if not 'bn' in n]
    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': bn_weight_decay},
                                 {'params': rest_params, 'weight_decay': weight_decay}],
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   verbose=False)

    return optimizer


def train_loop(args, model, criterion, optimizer, lr_scheduler, train_loader, train_loader_len, val_loader, epochs, fp16,
               use_amp=False, batch_size_multiplier=1,
               best_prec1=0, start_epoch=0,
               output_dir='./', train_log_freq=100, ckpt_freq=50,
               verbose=False, train_writer=None,
               layers=None, layer_structure=None,
               prune_target=None, prune_start=None, prune_end=None,
               method=0, prune_per_epoch=False, baseline_acc=None, pre_mask=None, wrap_solver=None, clipping=None):

    layer_bn, layer_gate, groups = layer_structure
    lookup_table, fmap_table = None, None
    pre_group, aft_group_list = None, None
    if method == 9 or method == 10 or method >= 20:  # latency aware pruning / latency targeted pruning
        from prune.prune_config_with_structure import PruneConfigReader
        config_reader = PruneConfigReader()
        config_reader.set_prune_setting(args.reg_conf.replace('.json', '_structure.json'))
        _, _, _, pre_group, aft_group_list = config_reader.get_layer_structure()

        latency_file_path = args.lut_file
        if latency_file_path is None or not os.path.exists(latency_file_path):
            print('Latency LUT needs to be provided! Exit.')
            exit(1)
        with open(latency_file_path, 'rb') as f:
            lookup_table = pkl.load(f)
        import json
        with open('net_structure/{}_fmap.json'.format(args.arch), 'r') as f:
            fmap_table = json.load(f)
        from prune.latency_targeted_pruning_mdp import load_group_size
        load_group_size(args.arch)
        print("Finished loading channel group size")
    metric_to_save = defaultdict(list)
    if prune_target is None:
        neuron_norm = defaultdict(list)
        neuron_norm = None
        layer_neuron_num = None
        layer_masks = None
    else:
        neuron_norm = None
        layer_neuron_num = defaultdict(list)
    layer_masks = {}
    # Initialize the masks
    for group in groups:
        for layer_name in group:
            layer_masks[layer_name] = torch.ones(layers[layer_name].weight.size(0)).to(layers[layer_name].weight.device)

    # if pre_mask is not None:
    #     assert os.path.exists(pre_mask)
    #     with open(pre_mask, 'rb') as f:
    #         layer_masks = pkl.load(f)
    #     # mask weights
    #     for group_name in layer_masks.keys():
    #         layer_masks[group_name] = layer_masks[group_name].cpu().cuda()
    #     for group_name, mask in layer_masks.items():
    #         for layer_name in group_name:
    #             layer = layers[layer_name]
    #             layer.weight.data.mul_(mask.view(-1, 1, 1, 1))
    #             if layer.bias is not None:
    #                 layer.bias.data.mul_(mask)
    #             # mask corresponding bn
    #             if layer_bn is not None:
    #                 bn_layer = layers[layer_bn[layer_name]]
    #                 bn_layer.weight.data.mul_(mask)
    #                 bn_layer.bias.data.mul_(mask)
    #                 bn_layer.running_mean.data.mul_(mask)
    #                 bn_layer.running_var.data.mul_(mask)

    bn_weights = []
    if method == 11:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                bn_weights.append(layer.weight)
    if method >= 20:
        from prune.latency_targeted_pruning_mdp import set_latency_prune_target, get_total_latency
        # Done Change
        # print(layer_masks)
        initial_latency = get_total_latency(layers, groups, pre_group, layer_masks, fmap_table, lookup_table, args.lut_bs)
        prune_target = set_latency_prune_target(initial_latency, 0, PRUNE_STEPS, args.target_latency, args.prune_ratio, mode=args.prune_mode)
        print(prune_target)

    pre_pruned_num, pre_target_num = 0, 0
    metric_outdir = os.path.join(output_dir, 'metrics')
    if verbose:
        if not os.path.exists(metric_outdir):
            os.makedirs(metric_outdir)

    for epoch_id in range(epochs):
        if verbose and baseline_acc is not None and epoch_id < len(baseline_acc):
            train_writer.add_scalars('accuracy', {'val/baseline': baseline_acc[epoch_id]}, epoch_id)
        if epoch_id < start_epoch:
            continue
        update_metric = (method not in [0, 1, 6] and prune_start <= epoch_id < prune_end)
        loss, neuron_metric_value = train(args, train_loader, train_loader_len,
                                          model, criterion,
                                          optimizer,
                                          lr_scheduler,
                                          fp16,
                                          epoch_id,
                                          use_amp=use_amp,
                                          batch_size_multiplier=batch_size_multiplier,
                                          train_log_freq=train_log_freq,
                                          layers=layers, groups=groups,
                                          layer_bn=layer_bn,
                                          layer_gate=layer_gate,
                                          method=method,
                                          layer_masks=layer_masks,
                                          update_metric=update_metric,
                                          bn_weights=bn_weights,
                                          prune_target=prune_target if not prune_per_epoch else None,
                                          prune_epoch=prune_start,
                                          output_dir=metric_outdir,
                                          pre_group=pre_group,
                                          aft_group_list=aft_group_list,
                                          step_size=args.step_size,
                                          mu=args.mu,
                                          fmap_table=fmap_table,
                                          lookup_table=lookup_table,
                                          wrap_solver=wrap_solver,
                                          clipping=clipping,
                                          lut_bs=args.lut_bs,
                                          pruned=args.pruned,
                                          pulp=args.pulp)

        prec1, prec5 = validate(val_loader, model, criterion)

        if verbose:
            train_writer.add_scalar('train/loss', loss, epoch_id)
            train_writer.add_scalar('train/LR', optimizer.param_groups[0]['lr'], epoch_id)
            train_writer.add_scalars('accuracy', {'val/before_prune': prec1}, epoch_id)

        ori_conv_neurons = count_non_zero_neurons(layers, layer_neuron_num, neuron_norm)
        if verbose:
            train_writer.add_scalar('neuron_num', ori_conv_neurons, epoch_id)
        for name, layer_metric in neuron_metric_value.items():
            metric_to_save[name] = layer_metric.cpu().numpy().squeeze()

        if verbose:
            if (epoch_id + 1) % ckpt_freq == 0:
                save_path = os.path.join(output_dir, 'epoch_{}.pth'.format(epoch_id + 1))
                torch.save(model.state_dict(), save_path)
                print('Saved model to {}.'.format(save_path))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                save_path = os.path.join(output_dir, 'best_net.pth')
                torch.save(model.state_dict(), save_path)
            if epoch_id == prune_end-1:
                save_path = os.path.join(output_dir, 'pruned_net_at{}.tar'.format(epoch_id))
                torch.save({
                    'epoch': epoch_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
                print('Saved pruned model and optimizer to {}.'.format(save_path))

            if neuron_norm is not None:
                with open('{}/layer_avg_norm.pkl'.format(output_dir), 'wb') as outfile:
                    pkl.dump(neuron_norm, outfile)
            if layer_neuron_num is not None:
                with open('{}/layer_neuron_num.pkl'.format(output_dir), 'wb') as outfile:
                    pkl.dump(layer_neuron_num, outfile)
            with open('{}/{}.pkl'.format(metric_outdir, epoch_id), 'wb') as outfile:
                pkl.dump(metric_to_save, outfile)

        if verbose:
            if layer_masks is not None:
                with open('{}/layer_masks.pkl'.format(output_dir), 'wb') as outfile:
                    pkl.dump({group_name: mask.cpu() for group_name, mask in layer_masks.items()}, outfile)
    return best_prec1


def get_hessian_step(model, input_var, target_var, criterion, optimizer, fp16, use_amp=False, bn_weights=None):
    output = model(input_var)
    loss = criterion(output, target_var)
    grad_w_p = autograd.grad(loss, bn_weights)
    grad_w = list(grad_w_p)

    N = input_var.size(0)
    inputs_one = [input_var[:N//2], input_var[N//2:]]
    targets_one = [target_var[:N//2], target_var[N//2:]]
    for cur_in, cur_tar in zip(inputs_one, targets_one):
        output = model(cur_in)
        loss = criterion(output, cur_tar)
        grad_f_p = autograd.grad(loss, bn_weights, create_graph=True)
        z = 0
        for count, grad_f in enumerate(grad_f_p):
            z += (grad_w[count].data * grad_f).sum()
        if torch.isnan(z):
            print("NAN in hessian calculation, skipping the batch for hessian")
            return
        if torch.distributed.is_initialized():
            reduced_z = reduce_tensor(z.data)
        else:
            reduced_z = loss.data
        if fp16:
            optimizer.backward(z)
        elif use_amp:
            with amp.scale_loss(z, optimizer) as scaled_z:
                scaled_z.backward()
        else:
            z.backward()
        for w in bn_weights:
            if torch.isnan(w.grad.data.mean()):
                optimizer.zero_grad()
                break


def get_train_step(model, criterion, optimizer, fp16, use_amp=False,
                   layers=None, groups=None, layer_bn=None, layer_gate=None, method=0,
                   neuron_metric=None, layer_masks=None, get_hessian=False, bn_weights=None,
                   pre_group=None, aft_group_list=None, step_size=8, mu=6e-4, fmap_table=None, lookup_table=None, clipping=None):
    def _step(input, target, optimizer_step=True, update_metric=False, last_iter=False, imp_convk=False):
        optimizer.zero_grad()

        input_var = Variable(input)
        target_var = Variable(target)

        if get_hessian:
            get_hessian_step(model, input_var, target_var, criterion, optimizer, fp16, use_amp, bn_weights)
            update_neuron_metric(layers, groups, layer_bn, layer_gate, method, neuron_metric)
            update_metric = False
            optimizer.zero_grad()

        output = model(input_var)
        loss = criterion(output, target_var)

        if torch.isnan(loss):
            print("NAN in loss, skipping the batch and update")
            reduced_loss, prec1, prec5 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            return reduced_loss, prec1, prec5

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

        has_nan = False
        for w in model.parameters():
            if torch.isnan(w.grad.data.mean()):
                has_nan = True
                break
        if has_nan:
            print("NAN in grad, skipping the batch and update")
            reduced_loss, prec1, prec5 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            return reduced_loss, prec1, prec5

        # mask gradients and gradient momentum buffer
        # Done change
        if layer_masks is not None:
            for layer_name, mask in layer_masks.items():
                weight = layers[layer_name].weight
                # assert weight.grad.data[mask==0.].sum().item() == 0.
                weight.grad.data.mul_(mask.view(-1, 1, 1, 1))
                weight.data.mul_(mask.view(-1, 1, 1, 1))
                param_state = optimizer.state[weight]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].data.mul_(mask.view(-1, 1, 1, 1))
                bias = layers[layer_name].bias
                if bias is not None:
                    bias.grad.data.mul_(mask)
                    bias.data.mul_(mask)
                    param_state = optimizer.state[bias]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].data.mul_(mask)
                if layer_bn is not None:
                    layers[layer_bn[layer_name]].weight.data.mul_(mask)
                    layers[layer_bn[layer_name]].weight.grad.data.mul_(mask)
                    layers[layer_bn[layer_name]].bias.data.mul_(mask)
                    layers[layer_bn[layer_name]].bias.grad.data.mul_(mask)
                    param_state = optimizer.state[layers[layer_bn[layer_name]].weight]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].data.mul_(mask)
                    param_state = optimizer.state[layers[layer_bn[layer_name]].bias]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].data.mul_(mask)

        if update_metric:
            # Done Change
            update_neuron_metric_mdp(layers, groups, layer_bn, layer_gate, method, neuron_metric,
                                 pre_group, aft_group_list, step_size, mu, layer_masks,
                                 fmap_table, lookup_table, last_iter, imp_convk)

        if optimizer_step:
            optimizer.step()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def train(args, train_loader, train_loader_len, model, criterion, optimizer, lr_scheduler, fp16, epoch_id,
          use_amp=False, batch_size_multiplier=1, train_log_freq=100,
          layers=None, groups=None, layer_bn=None, layer_gate=None, method=0, layer_masks=None,
          update_metric=False, bn_weights=None, prune_target=None, prune_epoch=0, output_dir='', prof=None,
          pre_group=None, aft_group_list=None, step_size=8, mu=0.0006, fmap_table=None, lookup_table=None,
          wrap_solver=None, clipping=None, lut_bs=256, pruned=False, pulp=False):
    """ Train one epoch """
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()
    compute_time = AverageMeter()
    ips = AverageMeter()

    model.train()
    neuron_metric_value = {}
    get_hessian = (method == 11 and update_metric)
    step = get_train_step(model, criterion, optimizer, fp16, use_amp=use_amp,
                          layers=layers, groups=groups,
                          layer_bn=layer_bn, layer_gate=layer_gate,
                          method=method, neuron_metric=neuron_metric_value,
                          layer_masks=layer_masks,
                          get_hessian=get_hessian, bn_weights=bn_weights,
                          pre_group=pre_group, aft_group_list=aft_group_list,
                          step_size=step_size, mu=mu, fmap_table=fmap_table, lookup_table=lookup_table, clipping=clipping)

    end = time.time()
    optimizer.zero_grad()
    blocks_num = args.blocks_num
    pre_pruned_num, pre_target_num = 0, 0
    batch_num = 0
    prune_counter = 0
    for i, (input, target) in enumerate(train_loader):
        if prof is not None and i >= prof:  # run only n iterations
            break
        lr_scheduler(optimizer, i, epoch_id)
        data_load_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, prec1, prec5 = step(input, target, optimizer_step=optimizer_step, update_metric=update_metric, last_iter=(i==train_loader_len-1), imp_convk=args.imp_convk)

        it_time = time.time() - end

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item())
        top5.update(prec5.item())
        data_time.update(data_load_time/input.size(0)*1000)
        compute_time.update((it_time-data_load_time)/input.size(0)*1000)
        ips.update(calc_ips(input.size(0), it_time - data_load_time))

        if (i+1) % train_log_freq == 0:
            print('Epoch {}, Iter {}, loss {}, top1 {}, top5 {}, data_time(ms/image) {}, compute_time(ms/image) {}, ips {}'.format(
                epoch_id, i+1, losses.avg, top1.avg, top5.avg, data_time.avg, compute_time.avg, ips.avg))
        batch_num += 1
        # prune every 40 iterations, finish in 30 steps (1200 iters) in total for imagenet
        if epoch_id == prune_epoch and prune_target is not None and (i+1)%ACCUM==0 and i<=1200:
        # if epoch_id == prune_epoch and prune_target is not None and (i+1)%40==0 and i<=1200:
            prune_counter += 1
            # Skip pruning if load a pruned model
            if pruned:
                continue
            if method in [0, 1, 6]:
                update_neuron_metric(layers, groups, layer_bn, layer_gate, method, neuron_metric_value)
            else:
                for group_name in neuron_metric_value.keys():
                    neuron_metric_value[group_name] = torch.div(neuron_metric_value[group_name], ACCUM)
            blocks = build_block_from_layers(layers)
            
            if blocks_num is not None:
                # blocks_num = (prune_counter > 0, blocks_num)
                blocks_num = (prune_counter > 25, blocks_num)
            total_pruned_num = apply_latency_target_pruning(output_dir, layers, layer_bn, blocks, layer_gate, neuron_metric_value, 
                                                            prune_target[i//ACCUM], layer_masks,
                                                            groups, pre_group, aft_group_list, fmap_table, lookup_table, method,
                                                            wrap_solver=wrap_solver, mu=mu, step_size=step_size,
                                                            lut_bs=lut_bs, pulp=pulp, blocks_num=blocks_num, no_blockprune=args.no_blockprune)

            to_save = {kk: vv.cpu().numpy() for kk, vv in neuron_metric_value.items()}
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                with open('{}/iter{}.pkl'.format(output_dir, i), 'wb') as f:
                    pkl.dump(to_save, f)

            cur_conv_neurons = count_non_zero_neurons(layers)
            print(total_pruned_num, cur_conv_neurons)
            for k, v in neuron_metric_value.items():
                neuron_metric_value[k][:] = 0
            batch_num = 0

        end = time.time()

    if method is not None:  # None is for fwd bwd time measure
        if method in [0, 1, 6]:
            update_neuron_metric(layers, groups, layer_bn, layer_gate, method, neuron_metric_value)
        else:
            for group_name in neuron_metric_value.keys():
                neuron_metric_value[group_name] = torch.div(neuron_metric_value[group_name], batch_num)

    print('Epoch {} Train ===== loss {}, top1 {}, top5 {}, data_time {}, compute_time {}, ips {}'.format(
        epoch_id, losses.avg, top1.avg, top5.avg, data_time.avg, compute_time.avg, ips.avg))

    return losses.avg, neuron_metric_value


def get_val_step(model, criterion):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader, model, criterion, prof=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()
    compute_time = AverageMeter()
    ips = AverageMeter()

    model.eval()
    step = get_val_step(model, criterion)

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        if prof is not None and i >= prof:  # run only n iterations
            break
        data_load_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item())
        top5.update(prec5.item())
        data_time.update(data_load_time/input.size(0)*1000)
        compute_time.update((it_time - data_load_time)/input.size(0)*1000)
        ips.update(calc_ips(input.size(0), it_time - data_load_time))

        end = time.time()

    print('Validate ===== loss {}, top1 {}, top5 {}, data_time(ms/image) {}, compute_time(ms/image) {}, ips {}'.format(
        losses.avg, top1.avg, top5.avg, data_time.avg, compute_time.avg, ips.avg))

    return top1.avg, top5.avg


def calc_ips(batch_size, time):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs/time
