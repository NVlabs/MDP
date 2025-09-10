"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import pdb

import sys

import torch.nn as nn

import numpy as np
import pdb

# from .pruning_engine_general import PruningConfigReader, pytorch_pruning
# from .pruning_engine_general_playground import PruningConfigReader, pytorch_pruning
from .pruning_engine_general_minlp import PruningConfigReader, pytorch_pruning
# from .pruning_engine_general_minlp_flops import PruningConfigReader, pytorch_pruning


def pruning_initialize_setup_rn50v15(args, model, log_save_folder, train_writer, parameters_for_update_named):
    pruning_engine = None
    if args.pruning:
        pruning_settings = dict()
        if not (args.pruning_config is None):
            pruning_settings_reader = PruningConfigReader()
            pruning_settings_reader.read_config(args.pruning_config)
            pruning_settings = pruning_settings_reader.get_parameters()

        pruning_parameters_list = prepare_pruning_list_resnet50v15(pruning_settings, model, model_name=args.arch,
                                                       pruning_mask_from="", name="withskip", verbouse=True)
        print("Total pruning layers:", len(pruning_parameters_list))

        folder_to_write = "%s"%log_save_folder+"/"
        log_folder = folder_to_write

        pruning_engine = pytorch_pruning(pruning_parameters_list, pruning_settings=pruning_settings, log_folder=log_folder)

        pruning_engine.connect_tensorboard(train_writer)
        pruning_engine.dataset = "ImageNet"
        pruning_engine.model = args.arch
        pruning_engine.pruning_mask_from = ""
        pruning_engine.load_mask()
        # gates_to_params = connect_gates_with_parameters_for_flops(args.arch, parameters_for_update_named)
        # pruning_engine.gates_to_params = gates_to_params

        # rewrite arguments from args string
    return pruning_engine

def prepare_pruning_list_resnet50v15(pruning_settings, model, model_name, pruning_mask_from='', name='', verbouse = True):
    '''
    Function returns a list of parameters from model to be considered for pruning.
    This method focuses on BN layer pruning, if string name contains withskip inside then it will consider pruning skip connections as well
    '''

    # pruning_method = pruning_settings['method']

    pruning_parameters_list = list()
    if verbouse:
        print("network structure")
    for module_indx, (m_name, m_el) in enumerate(model.named_modules()):
        # if isinstance(m_el, torch.nn.Conv2d):
        #     print(module_indx, m_name, m_el)
        # if isinstance(m_el, torch.nn.Linear):
        #     print(module_indx, m_name, m_el, m_el.weight.shape)
        if isinstance(m_el, torch.nn.BatchNorm2d):
            set_to_zero = []
            if verbouse:
                print(module_indx, m_name, m_el)

            layer = m_el

            #skipping the first conv layer
            # if ".layer" not in m_name:
            #     continue

            if "withskip" not in name:
                #no residual connection pruning
                if ".bn3" in m_name:
                    continue

                #no downsampling layer for now - later with skip connections
                if "downsample" in m_name:
                    continue

            #adding first conv layer 3->64
            #will be residual layer as well
            if 1:
                if m_name == "module.1.bn1" or m_name == "1.bn1":
                    prev_conv_layer = m_name.replace(".bn1", ".conv1")
                    next_layer = m_name.replace(".bn1", ".layer1.0.")

                    compute_criteria_from = [{"parameter_name": m_name + ".weight", "dim": 0}]
                    compute_criteria_from.append({"parameter_name": m_name + ".bias", "dim": 0})

                    set_to_zero = [{"parameter_name": m_name + ".weight", "dim": 0}]
                    set_to_zero.append({"parameter_name": m_name + ".bias", "dim": 0})
                    set_to_zero.append({"parameter_name": prev_conv_layer + ".weight", "dim": 0})
                    set_to_zero.append({"parameter_name": next_layer + "conv1" + ".weight", "dim": 1})
                    set_to_zero.append({"parameter_name": next_layer + "downsample.0" + ".weight", "dim": 1})

            # if m_name == "module.1.layer1.0.downsample.1":
            if ".downsample." in m_name:
                #working on skip connections
                # if m_name != "module.1.layer1.0.downsample.1":
                #     continue
                prefix = ""
                if "module." in m_name:
                    prefix = "module."
                    if "1." == m_name[:2]:
                        layer_id = int(m_name[len("module.1.layer"):len("module.1.layer")+1])
                        prefix = prefix + "1."
                    else:
                        layer_id = int(m_name[len("module.layer"):len("module.layer")+1])

                else:
                    if "1." == m_name[:2]:
                        layer_id = int(m_name[len("1.layer"):len("1.layer") + 1])
                        prefix = prefix + "1."
                    else:
                        layer_id = int(m_name[len("layer"):len("layer") + 1])


                blocks = [3, 4, 6, 3]
                #bn layer for the first skip connection
                compute_criteria_from = [{"parameter_name": m_name + ".weight", "dim": 0}]
                compute_criteria_from.append({"parameter_name": m_name + ".bias", "dim": 0})
                for layer_indx in range(blocks[layer_id-1]):
                    compute_criteria_from.append({"parameter_name": "%slayer%d.%d.bn3"%(prefix,layer_id,layer_indx) + ".weight", "dim": 0})
                    compute_criteria_from.append({"parameter_name": "%slayer%d.%d.bn3"%(prefix,layer_id,layer_indx) + ".bias", "dim": 0})

                set_to_zero = [{"parameter_name": m_name + ".weight", "dim": 0}]
                set_to_zero.append({"parameter_name": m_name + ".bias", "dim": 0})
                set_to_zero.append({"parameter_name": m_name.replace(".downsample.1",".conv3") + ".weight", "dim": 0})

                set_to_zero.append({"parameter_name": m_name.replace(".downsample.1",".downsample.0") + ".weight", "dim": 0})

                if layer_id < 4:
                    set_to_zero.append({"parameter_name": "%slayer%d.0.conv1"%(prefix,layer_id+1) + ".weight", "dim": 1})
                    set_to_zero.append({"parameter_name": "%slayer%d.0.downsample.0"%(prefix,layer_id+1) + ".weight", "dim": 1})
                if layer_id == 4:
                    #last skip connection
                    set_to_zero.append(
                        {"parameter_name": "%sfc" % (prefix, ) + ".weight", "dim": 1})
                for layer_indx in range(0, blocks[layer_id-1]):
                    set_to_zero.append({"parameter_name": "%slayer%d.%d.bn3"%(prefix,layer_id,layer_indx) + ".weight", "dim": 0})
                    set_to_zero.append({"parameter_name": "%slayer%d.%d.bn3"%(prefix,layer_id,layer_indx)  + ".bias", "dim": 0})
                    if layer_indx > 0:
                        set_to_zero.append(
                            {"parameter_name": "%slayer%d.%d.conv3" % (prefix, layer_id, layer_indx) + ".weight",
                             "dim": 0})
                        set_to_zero.append({"parameter_name": "%slayer%d.%d.conv1" % (prefix,layer_id,layer_indx)  + ".weight", "dim": 1})

            # no residual connection pruning, we added them earlier if wanted
            if ".bn3" in m_name:
                continue

            if "layer" in m_name and "downsample" not in m_name:
                if m_name.endswith(".bn1"):
                    prev_conv_layer = m_name.replace(".bn1",".conv1")
                    next_conv_layer = m_name.replace(".bn1", ".conv2")

                if m_name.endswith(".bn2"):
                    prev_conv_layer = m_name.replace(".bn2",".conv2")
                    next_conv_layer = m_name.replace(".bn2", ".conv3")

                # score will be computed from this element
                compute_criteria_from = [{"parameter_name": m_name+".weight", "dim": 0}]
                compute_criteria_from.append({"parameter_name": m_name+".bias", "dim": 0})


                #parameters that we will set to zero if decide to prune the one from which we computed the score
                set_to_zero = [{"parameter_name": m_name+".weight", "dim": 0}]
                set_to_zero.append({"parameter_name": m_name+".bias", "dim": 0})
                set_to_zero.append({"parameter_name": prev_conv_layer+".weight", "dim": 0}) #prev layer is affected by the input dim 1
                set_to_zero.append({"parameter_name": next_conv_layer+".weight", "dim": 1}) #next layer is affected by the input dim 1

                if verbouse:
                    print(m_name, prev_conv_layer, next_conv_layer)

            # print(len(set_to_zero))
            if len(set_to_zero) == 0:
                continue

            # change names of parameters to references of parameters
            for m_name, m_el in model.named_parameters():
                for el in set_to_zero:
                    if m_name == el["parameter_name"]:
                        el["parameter"] = m_el

                for indx in range(len(compute_criteria_from)):
                    if m_name == compute_criteria_from[indx]["parameter_name"]:
                        compute_criteria_from[indx]["parameter"] = m_el

            for_pruning = {"set_to_zero": set_to_zero, "layer": layer,
                           "compute_criteria_from": compute_criteria_from}
            if 0:
                #print structure to prune
                for el in set_to_zero:
                    print(el["parameter_name"], el["dim"])
                    print(el["parameter"].shape)

            pruning_parameters_list.append(for_pruning)

    return pruning_parameters_list



def dynamic_network_change_local_vialist(model, model_name, salient=False):
    '''
    Methods attempts to modify network in place by removing pruned filters.
    Works with ResNet101 for now only
    :param model: reference to torch model to be modified
    :return:
    '''
    def add_input_channel(param, non_zero_indx):
        if hasattr(param, "keep_channels_input"):
            param.keep_channels_input.extend(non_zero_indx)
        else:
            param.keep_channels_input = list()
            param.keep_channels_input.extend(non_zero_indx)

    def add_output_channel(param, non_zero_indx):
        if hasattr(param, "keep_channels_output"):
            param.keep_channels_output.extend(non_zero_indx)
        else:
            param.keep_channels_output = list()
            param.keep_channels_output.extend(non_zero_indx)

    def check_allow_trim(dict):
        res = False
        if "allow_trim" in dict.keys():
            if dict["allow_trim"]:
                res = True
        return res


    # change network dynamically given a pruning mask

    # step 1: model adjustment
    # lets go layer by layer and get the mask if we have parameter in pruning settings:
    if not salient:
        print("-------------Trimming the model-------------")
        print("=============Before compressing=============")
        print("printing conv layers")
        for module_indx, (m_name ,m) in enumerate(model.named_modules()):
            if "Conv" in m.__class__.__name__:
                if hasattr(m, "weight"):
                    print(module_indx, "->", m_name, "->", m.weight.data.shape)

    #Go over layers and replace pruned parameters

    addopted_layers = 0

    for layer_id, layer_info in enumerate(model.named_modules()):

        layer_name, layer_params = layer_info
        if not salient:
            print(layer_name)

        if not hasattr(layer_params, "do_pruning"):
            continue

        # print(".")

        if not layer_params.do_pruning:
            continue

        compute_criteria_from = layer_params.compute_criteria_from
        set_to_zero = layer_params.set_to_zero

        for w_ind, w in enumerate(compute_criteria_from):

            value = w["parameter"]
            nunits =  w["parameter"].shape[0]
            new_criteria = value.data.pow(2).view(nunits, -1).sum(dim=1)
            # if
            if w_ind==0:
                criteria_for_layer = new_criteria
            else:
                criteria_for_layer += new_criteria

        non_zero_indx = criteria_for_layer.nonzero().view(-1)

        for param in set_to_zero:
            if "shift" in param.keys():
                shift = param["shift"]
            else:
                shift = 0

            list_indeces_to_add = non_zero_indx + shift

            if check_allow_trim(param):
                # import pdb; pdb.set_trace()
                in_the_range = non_zero_indx + param["shift"] < param["parameter"].data.shape[param["dim"]]
                in_the_range = in_the_range*(non_zero_indx + param["shift"] >= 0)
                list_indeces_to_add = list_indeces_to_add[in_the_range]

            if param["dim"] == 0:
                #keep_channels_output
                # print(param["parameter_name"])
                add_input_channel(param["parameter"], list_indeces_to_add)
            elif param["dim"] == 1:
                add_output_channel(param["parameter"], list_indeces_to_add)

            # print(param["parameter_name"], shift)

        addopted_layers += 1


    ##second pass to keep only channels that are needed
    for layer_id, (layer_name, layer_params) in enumerate(model.named_parameters()):
        if hasattr(layer_params, "keep_channels_input"):
            if len(layer_params.keep_channels_input)>0:
                if hasattr(layer_params.keep_channels_input[0], "data"):
                    layer_params.keep_channels_input = [a.item() for a in layer_params.keep_channels_input]

            layer_params.keep_channels_input = np.asarray(layer_params.keep_channels_input)
            # print("changing input", layer_name, layer_params.keep_channels_input)
            # non_zero_indx = layer_params.keep_channels_input
            # after Sept15 2020
            non_zero_indx = np.unique(layer_params.keep_channels_input)
            layer_params.data = layer_params.data[non_zero_indx, ...]
            # layer_params = nn.Parameter(layer_params.data[non_zero_indx, ...])

        if hasattr(layer_params, "keep_channels_output"):
            if len(layer_params.keep_channels_output) > 0:
                if hasattr(layer_params.keep_channels_output[0], "data"):
                    layer_params.keep_channels_output = [a.item() for a in layer_params.keep_channels_output]
            layer_params.keep_channels_output = np.asarray(layer_params.keep_channels_output)
            # print("changing output", layer_name, layer_params.keep_channels_output)
            # non_zero_indx = layer_params.keep_channels_output
            # non_zero_indx = layer_params.keep_channels_output
            # after Sept15 2020

            non_zero_indx = np.unique(layer_params.keep_channels_output)
            # layer_params = nn.Parameter(layer_params.data[:, non_zero_indx])
            layer_params.data = layer_params.data[:, non_zero_indx]

            # try:
            #     layer_params.data = layer_params.data[:, non_zero_indx]
            # except:
            #     print(layer_name, layer_params.shape)





    ##address grouped convolutions:
    ##third pass to update group convolutions:
    ##also for batch norms
    for layer_id, (layer_name, layer_params) in enumerate(model.named_modules()):
        if hasattr(layer_params, "groups"):
            keep_inputs = hasattr(layer_params.weight, 'keep_channels_input')
            keep_outputs = hasattr(layer_params.weight, 'keep_channels_output')
            # print(f"Found layer {layer_name} that has groups of size {layer_params.groups}")
            # print(f"Weight shape is {layer_params.weight.shape}, "
            #       f"it has keep_dims: {keep_inputs}/{keep_outputs}")

            if layer_params.groups>1 and (keep_inputs or keep_outputs):
                if keep_inputs:
                    keep_inputs_neurons =  len(layer_params.weight.keep_channels_input)
                    # print(f"Changing groups from {layer_params.groups} to {keep_inputs_neurons}")
                    layer_params.groups = keep_inputs_neurons

                #means we are pruning
                # import pdb; pdb.set_trace()

        if "BatchNorm" in layer_params.__class__.__name__:
            #import pdb;pdb.set_trace()
            channels_keep = None
            keep_inputs = hasattr(layer_params.weight, 'keep_channels_input')
            keep_outputs = hasattr(layer_params.weight, 'keep_channels_output')

            if keep_inputs:
                # channels_keep = layer_params.weight.keep_channels_input
                #after sept15 2020:
                channels_keep = np.unique(layer_params.weight.keep_channels_input)
            if keep_outputs:
                # channels_keep = layer_params.weight.keep_channels_output
                #after sept15 2020:
                channels_keep = np.unique(layer_params.weight.keep_channels_output)
            if channels_keep is not None:
                # layer_params.running_mean = nn.Parameter(layer_params.running_mean.data[channels_keep])
                # layer_params.running_var = nn.Parameter(layer_params.running_var.data[channels_keep])
                layer_params.running_mean.data = layer_params.running_mean.data[channels_keep]
                layer_params.running_var.data = layer_params.running_var.data[channels_keep]

            # print(f"batch norm {layer_name}, weight: {layer_params.weight.shape}, mean: {layer_params.running_mean.shape}")

    if not salient:
        print("=============After compressing=============")
        print("printing conv layers")
        for module_indx, (m_name, m) in enumerate(model.named_modules()):
            if "Conv" in m.__class__.__name__:
                if hasattr(m, "weight"):
                    print(module_indx, "->", m_name, "->", m.weight.data.shape)

        # print("printing bn layers")
        for module_indx, m in enumerate(model.modules()):
            if "BatchNorm" in m.__class__.__name__:
                print(module_indx, "->", m.weight.data.shape)

        # print("printing gate layers")
        for module_indx, m in enumerate(model.modules()):
            if hasattr(m, "do_not_update"):
                print(module_indx, "->", m.weight.data.shape, m.size_mask)

def pruning_initialize_setup(log_save_folder, train_writer, pruning_parameters_list, pruning = True, pruning_config=None, tensorboard=True, latency_regularization=0.,latency_target=0., latency_look_up_table="", pruning_steps=50):
    pruning_engine = None
    if pruning:
        pruning_settings = dict()
        if not (pruning_config is None):
            pruning_settings_reader = PruningConfigReader()
            pruning_settings_reader.read_config(pruning_config)
            pruning_settings = pruning_settings_reader.get_parameters()

        print("Total pruning layers:", len(pruning_parameters_list))

        folder_to_write = "%s"%log_save_folder+"/"
        log_folder = folder_to_write

        pruning_engine = pytorch_pruning(pruning_parameters_list, pruning_settings=pruning_settings, log_folder=log_folder, latency_regularization=latency_regularization,latency_target=latency_target, latency_look_up_table=latency_look_up_table, pruning_steps=pruning_steps)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
            if tensorboard:
                from tensorboardX import SummaryWriter
                try:
                    train_writer = SummaryWriter(log_dir="%s" % (log_save_folder))
                except:
                    train_writer = SummaryWriter(logdir="%s" % (log_save_folder))

        pruning_engine.connect_tensorboard(train_writer)
        pruning_engine.dataset = "ImageNet"
        pruning_engine.model = "jasper"
        pruning_engine.pruning_mask_from = ""
        pruning_engine.load_mask()
        # gates_to_params = connect_gates_with_parameters_for_flops(args.arch, parameters_for_update_named)
        # pruning_engine.gates_to_params = gates_to_params

        # rewrite arguments from args string
    return pruning_engine


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        bufsize = 1
        self.log = open(filename, "w", buffering=bufsize)

    def delink(self):
        self.log.close()
        self.log = open('foo', "w")
#        self.write = self.writeTerminalOnly

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def prepare_logging(log_save_folder):
    # function initializing logging by putting a hook to stdout and stores data to the text file.
    # has a check for distriubted training to have only 1 logger.

    if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
        ## prepare log
        if not os.path.exists(log_save_folder):
            os.makedirs(log_save_folder)

        if not os.path.exists("%s/models" % (log_save_folder)):
            os.makedirs("%s/models" % (log_save_folder))
        if not os.path.exists("%s/images" % (log_save_folder)):
            os.makedirs("%s/images" % (log_save_folder))

        time_point = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        textfile = "%s/log_%s.txt" % (log_save_folder, time_point)
        stdout = Logger(textfile)
        sys.stdout = stdout
        print(" ".join(sys.argv))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        return lr

def set_lr(optimizer, value):
    for param_group in optimizer.param_groups:
        param_group['lr'] = value


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        # print("new  lr", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        # print("new  lr", epoch, warmup_length, epoch)
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def get_pruning_list_from_modules(named_parameters, verbouse=False, layer_type = -1):
    pruning_parameters_list = []
    if layer_type==-1:
        for module_indx, (m_name, m_el) in enumerate(named_parameters):
            if hasattr(m_el, "do_pruning"):
                if not m_el.do_pruning:
                    continue
                set_to_zero = m_el.set_to_zero
                compute_criteria_from = m_el.compute_criteria_from

                for_pruning = {"set_to_zero": set_to_zero, "layer": m_el,
                               "compute_criteria_from": compute_criteria_from, "name": m_name}
                if verbouse:
                    print(m_name, for_pruning["compute_criteria_from"][0]["parameter"].shape)

                if 0:
                    print("==========")
                    # print(m_name)
                    if len(m_el.compute_criteria_from) > 1:
                        for a in m_el.compute_criteria_from:
                            print(m_name, a["parameter"].shape)
                    print("----------")

                pruning_parameters_list.append(for_pruning)
    else:
        # assume the layer we want to prune is of type layer_type
        for module_indx, (m_name, m_el) in enumerate(named_parameters):
            if isinstance(m_el, layer_type):

                if verbouse:
                    print(module_indx, m_name, m_el, m_el.weight.shape)

                layer = m_el

                set_to_zero = [{"parameter_name": m_name + ".weight", "dim": 0, "parameter": m_el.weight}]

                compute_criteria_from = [{"parameter_name": m_name + ".weight", "dim": 0, "parameter": m_el.weight}]

                # pdb.set_trace()
                if hasattr(m_el, "bias"):
                    set_to_zero.append({"parameter_name": m_name + ".bias", "dim": 0, "parameter": m_el.bias})
                    compute_criteria_from.append({"parameter_name": m_name + ".bias", "dim": 0, "parameter": m_el.bias})

                for_pruning = {"set_to_zero": set_to_zero, "layer": layer,
                               "compute_criteria_from": compute_criteria_from}

                pruning_parameters_list.append(for_pruning)


    #for safety let's remove the last one
    #del pruning_parameters_list[-1]

    return pruning_parameters_list


def restore_from(model, path, local_rank=0, use_trick=True):
    def load_model_pytorch(model, load_model, model_name='resnet', gpu_n=0):

        print("=> loading checkpoint '{}'".format(load_model))

        SHOULD_I_PRINT = not torch.distributed.is_initialized() or torch.distributed.get_rank()==0

        if 1:
            checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage.cuda(gpu_n))

        if 1:
            if 'state_dict' in checkpoint.keys():
                load_from = checkpoint['state_dict']
            else:
                load_from = checkpoint

        # match_dictionaries, useful if loading model without gate:
        if 1:
            if 'module.' in list(model.state_dict().keys())[0]:
                if 'module.' not in list(load_from.keys())[0]:
                    from collections import OrderedDict

                    load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

            if 'module.' not in list(model.state_dict().keys())[0]:
                if 'module.' in list(load_from.keys())[0]:
                    from collections import OrderedDict

                    load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

        if use_trick:
            #for Jasper only:
            from collections import OrderedDict
            load_from = OrderedDict([(k.replace(".conv.", ".").replace(".mconv.", ".conv."), v) for k, v in load_from.items()])

        if 0:
            if list(load_from.items())[0][0][:2]=="1." and list(model.state_dict().items())[0][0][:2]!="1.":
                load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])

        if SHOULD_I_PRINT and 1:
            for ind, (key, item) in enumerate(model.state_dict().items()):
                if ind > 10:
                    continue
                print(key, model.state_dict()[key].shape)

            print("*********")

            for ind, (key, item) in enumerate(load_from.items()):
                if ind > 10:
                    continue
                print(key, load_from[key].shape)

        if 1:
            for key, item in model.state_dict().items():
                # if we add gate that is not in the saved file
                if key not in load_from:
                    load_from[key] = item
                # if load pretrined model
                if load_from[key].shape!=item.shape:
                    load_from[key] = item

        model.load_state_dict(load_from, strict=False)

        # del checkpoint

        epoch_from = -1
        if 'epoch' in checkpoint.keys():
            epoch_from = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_model, epoch_from))

        # checkpoint = None

    # self._pt_module.load_state_dict(t.load(path))
    # model._device

    # self.load_state_dict(t.load(path, map_location=load_device))

    load_model_pytorch(model, path, model_name='resnet', gpu_n="cuda")


def initialize_pruning_engine(named_modules, pruning_config = "", exp_name = "ttemp", prune_by = "predefined", tensorboard=True, latency_regularization=0., latency_target=0., latency_look_up_table="", pruning_steps=30):
    named_modules_list = [(a, b) for a, b in named_modules]

    named_modules = named_modules_list

    # dump structure of the model:
    structure = [b for a, b in named_modules]

    verbouse = True

    ##what to prune?
    if prune_by == "predefined":
        # pruning parameters are predefined by network architecture
        pruning_parameters_list = get_pruning_list_from_modules(named_modules, verbouse)
    else:
        # assume eprune_by is a class type we want to prune
        pruning_parameters_list = get_pruning_list_from_modules(named_modules, verbouse, layer_type = prune_by)


    if verbouse:
        print("preparing for pruning")
        print("total parameters:", len(pruning_parameters_list))

    train_writer = None

    prepare_logging(exp_name)
    pruning_engine = pruning_initialize_setup(log_save_folder=exp_name, train_writer=train_writer,
                                                  pruning_parameters_list=pruning_parameters_list, pruning=True,
                                                  pruning_config=pruning_config,
                                              tensorboard=tensorboard,latency_regularization=latency_regularization,latency_target=latency_target,latency_look_up_table=latency_look_up_table,pruning_steps=pruning_steps)

    pruning_engine.do_after_step = False

    with open(exp_name + "/network_structure.txt", "w") as f:
        for s in structure:
            f.write(s.__repr__())

    return pruning_engine

def initilize_layer_pruning(layer, bias=True, dim=0, parameter_name=None):
    # initializes pruning of particular layer by specifying relevant fields.
    # Later, the pruner will iterate over modules and will prune it if sees .do_pruning = True
    # Parameters from which to compute criteria are stored in the field compute_criteria_from
    # Parameters that needs to be pruned are stored in set_to_zero
    # There are other ways to specify what to prune.
    layer.do_pruning = True
    if parameter_name is None:
        parameter_name = layer.__repr__()

    compute_criteria_from = [{"parameter_name": parameter_name+".weight", "dim": dim, "parameter": layer.weight}, ]

    set_to_zero = [{"parameter_name": parameter_name+".weight", "dim": dim, "parameter": layer.weight}]
    if hasattr(layer, 'bias'):
        set_to_zero.append({"parameter_name": parameter_name+".bias", "dim": dim, "parameter": layer.bias})

    layer.compute_criteria_from = compute_criteria_from
    layer.set_to_zero = set_to_zero


def connect_output_to_input(parent_layer, child_parameter, dim=0, bias = False, shift = 0, allow_trim=False, parameter_name=None):
    # connects pruning to zero out dependent child layers
    # parent_layer - main layer from which to compute statistics
    # child_parameter  - parameter to be zeroed depending on the loss from parent
    # dim - dimension to be affected, 0 - output channel, 1 - input channel
    # bias - set to zero bias as well
    # shift - how much we should shift the child channel number with respect to parent, if negative then will be applied if index >= abs(shift)
    # allow_trim - if children parameter is smaller shape than we ignore the rest
    if parameter_name is None:
        parameter_name = child_parameter.__repr__()
    parent_layer.set_to_zero.append({"parameter_name": parameter_name+".weight",
                                                      "dim": dim, "parameter": child_parameter.weight, "shift": shift, "allow_trim": allow_trim})

    if not isinstance(child_parameter, nn.ConvTranspose2d):
        ADD_BIAS = (hasattr(child_parameter, 'bias') and dim == 0)
        dim2 = 0
    else:
        ADD_BIAS = (hasattr(child_parameter, 'bias') and dim == 1)
        dim2 = 0


    if ADD_BIAS:
        # pdb.set_trace()
        if child_parameter.bias is not None:
            parent_layer.set_to_zero.append({"parameter_name": parameter_name+".bias",
                                                 "dim": dim2, "parameter": child_parameter.bias, "shift": shift, "allow_trim": allow_trim})

def link_criteria_layers(parent_parameter, child_parameter, dim = 0, bias = False):
    parent_parameter.compute_criteria_from.append({"parameter_name": child_parameter.__repr__(),
                                         "dim": dim, "parameter": child_parameter.weight})
    if hasattr(child_parameter, 'bias'):
        parent_parameter.compute_criteria_from.append({"parameter_name": child_parameter.__repr__(),
                                             "dim": dim, "parameter": child_parameter.bias})
