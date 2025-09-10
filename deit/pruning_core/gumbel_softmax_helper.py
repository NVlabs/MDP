"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import os

class GumbelSoftMax(object):
    def __init__(self, args):
        super(GumbelSoftMax, self).__init__()
        self.gumbel_gates = args.gumbel_gates
        self.act_weight = args.act_weight
        self.target_rate = args.target_rate
        self.lr_gate = args.lr_gate
        self.temperature = args.temperature
        self.gumbel_optimizer = args.gumbel_optimizer

        self.model = args.model

        if args.gate_parameters_for_update is not None:
            self.gate_parameters_for_update = args.gate_parameters_for_update

        if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
            folder_to_write = "%s"%args.name + "/"
            log_folder = folder_to_write
            folder_to_write_debug = log_folder + '/debug/'
            if not os.path.exists(folder_to_write_debug):
                os.makedirs(folder_to_write_debug)
            self.log_debug = folder_to_write_debug + 'debugOutput_pruning' + '.txt'
            with open(self.log_debug, 'w') as f:
                f.write("gumbel softmax trick")

        self.act_loss = 0.0

        self.stop_train_after_epochs = args.gumbel_stop_train_after_epochs

    def get_loss(self):
        gate_function = lambda x: torch.sigmoid(x)

        acts_indx = 0
        acts_var = 0

        cum_points = [0.05, 0.1, 0.5, 0.9, 0.95]
        cum_acc = [0, ] * len(cum_points)

        act_weights = 0
        for ge, gate in enumerate(self.gate_parameters_for_update):

            # if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
            #     print("pre",ge, gate_function(gate).mean().item())

            act_weights += torch.clamp(gate_function(gate), 0, 1).sum()
            acts_var += torch.clamp(gate_function(gate), 0, 1).var() / len(self.gate_parameters_for_update)
            acts_indx += gate.nelement()

            for ci, cum_interval in enumerate(cum_points):
                cum_acc[ci] += (gate_function(gate) < cum_interval).sum()

        act_weight_mean = act_weights / acts_indx

        self.cum_acc = cum_acc
        self.cum_points = cum_points

        # loss from weights
        acts = 1.0 - act_weight_mean

        target_rate = self.target_rate
        act_weight = self.act_weight

        # keep rate
        self.acts = acts

        act_loss = act_weight * abs(acts - target_rate)
        self.act_loss = act_loss

        return act_loss

    def do_step(self, epoch, args, optimizer, model):
        # performs update of gates and conditionally stops training gates
        if self.stop_train_after_epochs < epoch:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
                print("gumbel_softmax_helper: stop training gates because of stop_train_after_epochs")

            self.set_gates_fixed()
            args.gumbel_gates = False

            # set momentum to zero
            if args.model=="resnet20" or args.model=="resnet101" or args.dataset=="Imagenet":
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        param_state = optimizer.state[p]
                        if 'momentum_buffer' in param_state:
                            del param_state['momentum_buffer']

            self.log(epoch, -1)

            for module_indx, m in enumerate(model.modules()):
                if hasattr(m, "do_not_update"):
                    m.discrete = True

            return 0

        # torch.nn.utils.clip_grad_norm_(self.gate_parameters_for_update, 1.0)
        if 0:
            for gateid, gate in enumerate(self.gate_parameters_for_update):
                # import pdb; pdb.set_trace()
                if gateid>0:
                    continue
                if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
                    print(gateid, gate.grad.data.mean().item())

        if 0:
            print("before:", self.gate_parameters_for_update[0].data[:3])
        self.gumbel_optimizer.step()
        if 0:
            print("after_:", self.gate_parameters_for_update[0].data[:3])

    def log(self, epoch, batch_idx):
        if not (not torch.distributed.is_initialized() or torch.distributed.get_rank()==0):
            return -1

        def write_to_debug(where_to_write, what_write_name, what_write_value):
            ### Aux function to store information
            with open(where_to_write, 'a') as f:
                f.write("{} {}\n".format(what_write_name, what_write_value))

        gate_function = lambda x: torch.sigmoid(x)

        write_to_debug(self.log_debug, "epoch", epoch)
        write_to_debug(self.log_debug, "iter", batch_idx)

        total_above95 = 0
        total_above50 = 0
        total = 0

        for layer, gate in enumerate(self.gate_parameters_for_update):
            if 0:
                # import pdb; pdb.set_trace()
                if layer == 0:
                    print(layer, gate.data[:5])

            write_to_debug(self.log_debug, "\nLayer:", layer)
            write_to_debug(self.log_debug, "units:", gate.nelement())

            ### if act_weights is 1 then pruned
            # pruned: torch.clamp(gate_function(gate), 0, 1)
            pruned_5 = (torch.clamp(gate_function(gate), 0, 1) > 0.5).sum()
            pruned_95 = (torch.clamp(gate_function(gate), 0, 1) > 0.95).sum()

            # print(layer, torch.clamp(gate_function(gate), 0, 1).mean().item())
            # import pdb; pdb.set_trace()

            total_above50 += pruned_5
            total_above95 += pruned_95
            total += gate.nelement()

            write_to_debug(self.log_debug, "pruned_perc:",
                           [pruned_5.item(), gate.nelement()])

            write_to_debug(self.log_debug, "pruned_perc95:",
                           [pruned_95.item(), gate.nelement()])

        write_to_debug(self.log_debug, "\nover all layers:",
                       "total: {}, pruned with percent above 50: {}, above 95: {}\n".format(total, total_above50,
                                                                                            total_above95))
        for a_i, _ in enumerate(self.cum_acc):
            write_to_debug(self.log_debug, "cum_points",
                           "{}: {}".format(self.cum_points[a_i], self.cum_acc[a_i]))

        write_to_debug(self.log_debug, "\n\n\n", "")

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:

    def set_gates_fixed(self, threshold = 0.0, min_cap = -12, max_cap = 12):
        # go through parameters and set them to 0..1 based on threshold
        for gate in self.gate_parameters_for_update:
            # import pdb; pdb.set_trace()
            gate.data[gate.data  < threshold] = min_cap
            gate.data[gate.data >= threshold] = max_cap
            #gate.discrete = True
