import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
from apex import amp
from pruning_core.pruning_utils import lr_cosine_policy

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def finetune_one_epoch(args, student, teacher, masks, criterion, orig_criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,main_loss_coeff=100000,original_loss_coeff=0.):
    student.train(set_training_mode)
    teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('main_loss_value', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('orig_loss_value', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            # get student predictions
            student_logits = student(samples)
            
            if main_loss_coeff>0:
                with torch.no_grad():
                    teacher_logits = teacher(samples)
                # get distillation loss
                main_loss = (criterion(student_logits[0], teacher_logits[0])+criterion(student_logits[1], teacher_logits[1]))/2
            else:
                main_loss = 0.
            
            original_loss_student = orig_criterion(samples, student_logits, targets)

        #total loss
        loss = main_loss_coeff*main_loss + original_loss_coeff*original_loss_student
        loss_value = loss.item()
        main_loss_value = 0.
        if main_loss_coeff>0:
            main_loss_value = main_loss.item()
        orig_loss_value = original_loss_student.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=student.parameters(), create_graph=is_second_order)

        if masks is not None:
            for name, p in student.named_parameters():
                if 'weight' in name or 'bias' in name or 'mask' in name:
                    p.grad.data = p.grad.data*masks[name]
                    p.data = p.data*masks[name]

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(student)

        metric_logger.update(loss=loss_value)
        metric_logger.update(main_loss_value=main_loss_value)
        metric_logger.update(orig_loss_value=orig_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch(args, student, teacher, pruning_engine, criterion, orig_criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,main_loss_coeff=100000,original_loss_coeff=0.):
    student.train(set_training_mode)
    teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('main_loss_value', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('orig_loss_value', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                teacher_logits = teacher(samples)
                teacher_logit = (teacher_logits[0]+teacher_logits[1])/2

            # get student predictions
            student_logits = student(samples)
            
            if not isinstance(student_logits, torch.Tensor):
                # assume that the model outputs a tuple of [outputs, outputs_kd]
                student_logit = (student_logits[0]+student_logits[1])/2

            # get distillation loss
            main_loss = criterion(student_logit, teacher_logit)
            original_loss_student = orig_criterion(samples, student_logits, targets)

        #total loss
        loss = main_loss_coeff*main_loss + original_loss_coeff*original_loss_student
        loss_value = loss.item()
        main_loss_value = main_loss.item()
        orig_loss_value = original_loss_student.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=student.parameters(), create_graph=is_second_order)

        if args.pruning:
            # will set pruned weights to zero
            pruning_engine.do_step(loss.item(), optimizer=optimizer)


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(student)

        metric_logger.update(loss=loss_value)
        metric_logger.update(main_loss_value=main_loss_value)
        metric_logger.update(orig_loss_value=orig_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def prune_with_Taylor(args, start_epoch, start_iter, data_loader, student, teacher, pruning_engine,
                        optimizer_gates_lr, interval_prune, gate_loss_coeff,
                        main_loss_coeff, prune_neurons_max, optimizer, lr_scheduler, loss_scaler, bs, criterion, orig_criterion,
                        amp_enable=True, student_eval=False, prune_per_iteration = 128, original_loss_coeff = 0.0, mixup_fn=None):
    # Function will prune based on Taylor

    train_main = args.train_main
    ####### do pruning ###############


    student.train()

    if pruning_engine.pruning_iterations_done > 0:
        student.eval()
    else:
        if student_eval==True and original_loss_coeff == 0:
            print("Changing student_eval to False for the first pruning iteraion, otherwise loss is 0")
            student.train()
    t_coeff = main_loss_coeff
    main_loss_coeff = 0.

    teacher.eval()

    iter_num = 0

    feature_loss = nn.MSELoss().cuda()

    iter_num = 0

    epoch = 0
    data_loader.sampler.set_epoch(args.seed + 0)

    pruning_engine.prune_per_iteration = prune_per_iteration
    pruning_engine.prune_neurons_max += prune_neurons_max

    if pruning_engine.pruning_iterations_done == 0:
        pruning_engine.maximum_pruning_iterations += int(prune_neurons_max/pruning_engine.prune_per_iteration)+1
    else:
        pruning_engine.maximum_pruning_iterations += int(prune_neurons_max/pruning_engine.prune_per_iteration)

    pruning_engine.frequency = interval_prune 
    
    # 100 * (39000 / 32)
    iterations_total = interval_prune * (int(prune_neurons_max/pruning_engine.prune_per_iteration)+2)
    

    for epoch in range(0, args.epochs):
        print('working on epoch {} ...'.format(epoch + start_epoch + 1))
        print(optimizer.param_groups[0]["lr"])
        # Set the data loader epoch to shuffle the index iterator.
        data_loader.sampler.set_epoch(args.seed + epoch + start_epoch)


        # For all the batches in the dataset.
        if iter_num > iterations_total:
            break
        
        for samples, targets in data_loader:
            samples = samples.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            # check if done all of them
            iter_num += 1
            if iter_num > iterations_total:
                break

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    teacher_logits = teacher(samples)
                    teacher_logit = (teacher_logits[0]+teacher_logits[1])/2
    
                # get student predictions
                student_logits = student(samples)
                
                if not isinstance(student_logits, torch.Tensor):
                    # assume that the model outputs a tuple of [outputs, outputs_kd]
                    student_logit = (student_logits[0]+student_logits[1])/2
    
                # get distillation loss
                main_loss = criterion(student_logit, teacher_logit)
                original_loss_student = orig_criterion(samples, student_logits, targets)
    
            #total loss
            loss = main_loss_coeff*main_loss + original_loss_coeff*original_loss_student
            # loss = main_loss_coeff*main_loss

            if 1:
                # do training
                # do backward pass
                # backward_step(optimizer, student, loss, args, timers)

                # optimizer step
                # if not student_eval:
                #     optimizer.step()

                optimizer.zero_grad()

                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=student.parameters(), create_graph=is_second_order)

                status = pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)
                torch.cuda.synchronize()
                
                if status == -2:
                    print('Pruning done! Target latency reached.')
                    return epoch+start_epoch, iter_num+start_iter, True

                if pruning_engine.pruning_iterations_done > 0:
                    if student.training and student_eval:
                        print("Student train() to eval()")
                        student.eval()
                        main_loss_coeff = t_coeff
                        #next need to reset momentum otherwise loss will jump a lot
                        for layer, if_prune in enumerate(pruning_engine.prune_layers):
                            if not if_prune:
                                continue
                            if pruning_engine.use_momentum:
                                pruning_engine.prune_network_accomulate["averaged"][layer] *= 0.0



            if iter_num%30 == 0:
                print(
                    f"it {iter_num+start_iter}, loss {main_loss.item():.7f}")
                    
                if utils.is_main_process():
                    if pruning_engine.train_writer is not None:
                        pruning_engine.train_writer.add_scalar('dicrete_opt/loss', main_loss.item(),
                                                               iter_num+start_iter)
                        pruning_engine.train_writer.add_scalar('dicrete_opt/original_loss_student',
                                                               original_loss_student.item(),
                                                               iter_num+start_iter)
    
                    if pruning_engine.train_writer is not None:
                        pruning_engine.train_writer.add_scalar('dicrete_opt_weighted/loss', main_loss_coeff*main_loss.item(),
                                                               iter_num+start_iter)
                        pruning_engine.train_writer.add_scalar('dicrete_opt_weighted/total_loss', loss.item(),
                                                               iter_num+start_iter)
        if args.output_dir:
            checkpoint_paths = [args.output_dir+'/checkpoint.pth',args.output_dir+'/pruned_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': student.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch+start_epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        
        lr_scheduler.step(0)#+start_epoch)
    
    return epoch+start_epoch, iter_num+start_iter, False

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            x, x_dist = model(images)
            output = (x + x_dist) / 2
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
