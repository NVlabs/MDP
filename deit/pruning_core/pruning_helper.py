"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.optim.optimizer import Optimizer
import torch

PRINT_ALL = False
USE_FULL = False

class pruning_helper(Optimizer):
    r"""Implements helping functions to count number of FLOPS, number of parameters, remove biases of pruned weights etc.
    """

    def __init__(self, parameters=None, named_parameters=None, output_sizes=None):
        defaults = dict(total_neurons = 0)

        super(pruning_helper, self).__init__(parameters, defaults)

        self.per_layer_per_neuron_stats = {'flops': list(), 'params': list(), 'latency': list()}

        self.named_parameters = named_parameters

        self.output_sizes = None
        if output_sizes is not None:
            self.output_sizes = output_sizes

    def __setstate__(self, state):
        super(pruning_helper, self).__setstate__(state)

    def get_number_neurons(self):
        total_neurons = 0
        for gr_ind, group in enumerate(self.param_groups):
            total_neurons += group['total_neurons'].item()

        return total_neurons

    def get_number_flops(self, print_output = False):
        total_flops = 0
        for gr_ind, group in enumerate(self.param_groups):
            total_flops += group['total_flops'].item()

        total_neurons = 0
        for gr_ind, group in enumerate(self.param_groups):
            total_neurons += group['total_neurons'].item()

        if print_output:
            print("Flops 1e 9/params 1e7:  %3.3f &  %3.3f"%(total_flops/1e9, total_neurons/1e7))
        return total_flops

    def step_after(self):
        """Computes FLOPS and number of neurons after considering zeroed out input and outputs.
        Channels are assumed to be pruned if their l2 norm is very small or if magnitude of gradient is very small.
        This function does not perform weight pruning, weights are untouched. It just count number of small weights

        This function also calls push_biases_down which sets corresponding biases to 0.

        """


        param_index = -1
        conv_param_index = -1
        for group in self.param_groups:
            group['total_neurons'] = None
            group['total_flops'] = None

            for p in group['params']:

                if 1:
                    weight_size = p.data.size()

                    if (len(weight_size) == 4) or (len(weight_size) == 3) or (len(weight_size) == 2) or (len(weight_size) == 1):
                        param_index += 1
                        # defined for conv layers only
                        nunits = p.data.size(0)
                        if nunits == 0:
                            continue


                        # let's compute denominator
                        divider = p.data.pow(2).view(nunits,-1).sum(dim=1).pow(0.5)

                        eps = 1e-4
                        # check if divider is above threshold
                        divider_bool = divider.gt(eps).view(-1).type_as(p.data)

                        if (len(weight_size) == 4) or (len(weight_size) == 3) or (len(weight_size) == 2) or (len(weight_size) == 1):
                            if not (p.grad is None):
                                # consider gradients as well and if gradient is below spesific threshold than we claim parameter to be removed
                                divider_grad = p.grad.data.pow(2).view(nunits, -1).sum(dim=1).pow(0.5)
                                eps = 1e-8
                                divider_bool_grad = divider_grad.gt(eps).view(-1).type_as(p.data)
                                divider_bool = divider_bool_grad * divider_bool

                                if (len(weight_size) == 4) or (len(weight_size) == 3) or (len(weight_size) == 2):
                                    # get gradient for input:
                                    divider_grad_input = p.grad.data.pow(2).transpose(0,1).contiguous().view(p.data.size(1),-1).sum(dim=1).pow(0.5)
                                    divider_bool_grad_input = divider_grad_input.gt(eps).view(-1).type_as(p.data)

                                    divider_input = p.data.pow(2).transpose(0,1).contiguous().view(p.data.size(1), -1).sum(dim=1).pow(0.5)
                                    divider_bool_input = divider_input.gt(eps).view(-1).type_as(p.data)
                                    divider_bool_input = divider_bool_input * divider_bool_grad_input
                                    # if gradient is small then remove it out

                        if USE_FULL:
                            # reset to evaluate true number of flops and neurons
                            # useful for full network only
                            divider_bool = 0.0*divider_bool + 1.0
                            divider_bool_input = 0.0*divider_bool_input + 1.0

                        if len(weight_size) == 4:
                            # p.data.mul_(divider_bool.view(nunits,1, 1, 1).repeat(1,weight_size[1], weight_size[2], weight_size[3]))
                            current_neurons = divider_bool.sum()*divider_bool_input.sum()*weight_size[2]* weight_size[3]

                        if len(weight_size) == 3:
                            # For 1d convolution
                            current_neurons = divider_bool.sum()*divider_bool_input.sum()*weight_size[2]

                        if len(weight_size) == 2:
                            current_neurons = divider_bool.sum()*divider_bool_input.sum()

                        if len(weight_size) == 1:
                            current_neurons = divider_bool.sum()
                            # add mean and var over batches
                            current_neurons = current_neurons + divider_bool.sum()

                        # group['total_neurons'] += current_neurons.type_as(p.data)
                        if not (group['total_neurons'] is None):
                            group['total_neurons'] += current_neurons.type_as(group['total_neurons'])
                        else:
                            group['total_neurons'] = current_neurons

                        # print(len(weight_size), conv_param_index, len(self.output_sizes))
                        if len(weight_size) == 4:
                            conv_param_index += 1
                            input_channels  = divider_bool_input.sum()
                            output_channels = divider_bool.sum()

                            if self.output_sizes is not None:
                                output_height, output_width = self.output_sizes[conv_param_index][-2:]
                            else:
                                if hasattr(p, 'output_dims'):
                                    output_height, output_width = p.output_dims[-2:]
                                else:
                                    output_height, output_width = 0, 0

                            kernel_ops = weight_size[2] * weight_size[3] * input_channels

                            params = output_channels * kernel_ops
                            flops  = params * output_height * output_height

                            # add flops due to batch normalization
                            flops = flops + output_height * output_width*3

                        if len(weight_size) == 3:
                            conv_param_index += 1
                            input_channels  = divider_bool_input.sum()
                            output_channels = divider_bool.sum()

                            if self.output_sizes is not None:
                                output_height = self.output_sizes[conv_param_index][-1:]
                            else:
                                if hasattr(p, 'output_dims'):
                                    output_height = p.output_dims[-1:]
                                else:
                                    output_height = 0, 0
                            output_height = output_height[0]
                            kernel_ops = weight_size[2] * input_channels

                            params = output_channels * kernel_ops
                            flops  = params * output_height

                            # add flops due to batch normalization
                            flops = flops + output_height * 3

                        if len(weight_size) == 1:
                            flops = len(weight_size)
                        if len(weight_size) == 2:
                            input_channels  = divider_bool_input.sum()
                            output_channels = divider_bool.sum()
                            flops = input_channels * output_channels

                        if not (group['total_flops'] is None):
                            group['total_flops'] += flops
                        else:
                            group['total_flops'] = flops

                        if len(self.per_layer_per_neuron_stats['flops']) <= param_index:
                            self.per_layer_per_neuron_stats['flops'].append(flops / divider_bool.sum())
                            self.per_layer_per_neuron_stats['params'].append(current_neurons / divider_bool.sum())
                        else:
                            self.per_layer_per_neuron_stats['flops'][param_index] = flops / divider_bool.sum()
                            self.per_layer_per_neuron_stats['params'][param_index] = current_neurons / divider_bool.sum()

        # self.push_biases_down(eps=1e-6)

    def push_biases_down(self, eps=1e-3):
        '''
        This function goes over parameters and sets according biases to zero,
        without this function biases will not be zero
        '''
        # first pass
        list_of_names = []
        for name, param in self.named_parameters:
            if "weight" in name:
                weight_size = param.data.shape
                if (len(weight_size) == 4) or (len(weight_size) == 2):
                    # defined for conv layers only
                    nunits = weight_size[0]
                    # let's compute denominator
                    divider = param.data.pow(2).view(nunits, -1).sum(dim=1).pow(0.5)
                    divider_bool = divider.gt(eps).view(-1).type_as(param.data)
                    list_of_names.append((name.replace("weight", "bias"), divider_bool))

        # second pass
        for name, param in self.named_parameters:
            if "bias" in name:
                for ind in range(len(list_of_names)):
                    if list_of_names[ind][0] == name:
                        param.data.mul_(list_of_names[ind][1])

    def report_to_tensorboard(self, train_writer, global_iteration):
        if train_writer is None:
            return 0

        try:
            neurons_left = int(self.get_number_neurons())
            flops = int(self.get_number_flops())
        except:
            neurons_left = 1.0
            flops = 1.0

        train_writer.add_scalar('pruning/total_parameters_left', neurons_left, global_iteration)
        train_writer.add_scalar('pruning/total_flops_left', flops, global_iteration)


def add_hook_for_flops(model):
    # add output dims for FLOPs computation
    for module_indx, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            def forward_hook(self, input, output):
                self.weight.output_dims = output.shape

            m.register_forward_hook(forward_hook)

        if isinstance(m, torch.nn.Conv1d):
            def forward_hook(self, input, output):
                if isinstance(output, tuple):
                    self.weight.output_dims = output[0].shape
                else:
                    self.weight.output_dims = output.shape

            m.register_forward_hook(forward_hook)


def get_conv2d_sizes(model, data):
    output_sizes = None
    # add hooks to compute dimensions of the output tensors for conv layers
    add_hook_for_flops(model)
    if 1:
        # run inference
        with torch.no_grad():
            model(data)
        # store flops
        output_sizes = list()
        for param in model.parameters():
            if hasattr(param, 'output_dims'):
                output_dims = param.output_dims
                output_sizes.append(output_dims)

    # import pdb; pdb.set_trace()

    return output_sizes

