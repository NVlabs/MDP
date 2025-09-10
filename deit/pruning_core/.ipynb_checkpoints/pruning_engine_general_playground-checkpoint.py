"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from copy import deepcopy
import itertools
import pickle
import json
import pdb
import copy

# from utils import connect_gates_with_parameters_for_flops

METHOD_ENCODING = {0: "Taylor_weight", 1: "Random", 2: "Weight norm", 3: "Weight_abs",
                   6: "Taylor_output", 10: "OBD", 11: "Taylor_gate_SO",
                   22: "Taylor_gate", 23: "Taylor_gate_expectation", 30: "BN_weight", 31: "BN_Taylor"}


# Method is encoded as an integer that mapping is shown above.
# Methods map to the paper as follows:


##Works for 22- Taylor_gate only

# 0 - Taylor_weight - Conv weight/conv/linear weight with Taylor FO In Table 2 and Table 1
# 1 - Random        - Random
# 2 - Weight norm   - Weight magnitude/ weight
# 3 - Weight_abs    - Not used
# 6 - Taylor_output - Taylor-output as is [27]
# 10- OBD           - OBD
# 11- Taylor_gate_SO- Taylor SO
# 22- Taylor_gate   - Gate after BN in Table 2, Taylor FO in Table 1
# 30- BN_weight     - BN scale in Table 2
# 31- BN_Taylor     - BN scale Taylor FO in Table 2

LATENCY_COMPUTE = True
LATENCY_COMPUTE = True


def check_allow_trim(dict):
    res = False
    if "allow_trim" in dict.keys():
        if dict["allow_trim"]:
            res = True
    return res

class PruningConfigReader(object):
    def __init__(self):
        self.pruning_settings = {}
        self.config = None

    def read_config(self, filename):
        # reads .json file and sets values as pruning_settings for pruning
        # pdb.set_trace()
        with open(filename, "r") as f:
            self.config = json.load(f)

        if "jobConfiguration" in self.config.keys():
            self.config = self.config["jobConfiguration"]

        # self.config = config["jobConfiguration"]

        self.read_field_value("method", 0)
        self.read_field_value("frequency", 500)
        self.read_field_value("prune_per_iteration", 2)
        self.read_field_value("maximum_pruning_iterations", 10000)
        self.read_field_value("starting_neuron", 0)

        self.read_field_value("fixed_layer", -1)
        # self.read_field_value("use_momentum", False)

        self.read_field_value("pruning_threshold", 100)
        self.read_field_value("start_pruning_after_n_iterations", 0)
        # self.read_field_value("use_momentum", False)
        self.read_field_value("do_iterative_pruning", True)
        self.read_field_value("fixed_criteria", False)
        self.read_field_value("do_after_step", True)
        self.read_field_value("set_moment_zero", True)
        self.read_field_value("reset_gates", False)
        self.read_field_value("seed", 0)
        self.read_field_value("pruning_momentum", 0.9)
        self.read_field_value("flops_regularization", 0.0) #0.999
        self.read_field_value("params_regularization", 0.0) #0.9
        self.read_field_value("latency_regularization", 0.0) #0.9
        self.read_field_value("latency_look_up_table", "") #0.9
        self.read_field_value("prune_neurons_max", 1)

        self.read_field_value("group_size", 1)
        self.read_field_value("leave_at_least_one_group", True)
        self.read_field_value("push_down_weight_decay", 1e-3)


        self.read_field_value("push_down", False)

    def read_field_value(self, key, default):
        param = default
        if key in self.config:
            param = self.config[key]

        self.pruning_settings[key] = param

    def get_parameters(self):
        return self.pruning_settings


class pytorch_pruning(object):
    def __init__(self, for_pruning_parameters, pruning_settings=dict(), log_folder=None, latency_regularization=0., latency_target=0., latency_look_up_table=""):
        def initialize_parameter(object_name, settings, key, def_value):
            '''
            Function check if key is in the settings and sets it, otherwise puts default momentum
            :param object_name: reference to the object instance
            :param settings: dict of settings
            :param def_value: def value for the parameter to be putted into the field if it doesn't work
            :return:
            void
            '''
            value = def_value
            if key in settings.keys():
                value = settings[key]
            setattr(object_name, key, value)

        # store some statistics
        self.min_criteria_value = 1e6
        self.max_criteria_value = 0.0
        self.median_criteria_value = 0.0
        self.neuron_units = 0
        self.all_neuron_units = 0
        self.pruned_neurons = 0
        self.gradient_norm_final = 0.0
        self.flops_regularization = 0.0 #not used in the paper
        self.pruning_iterations_done = 0

        # initialize_parameter(self, pruning_settings, 'use_momentum', False)
        initialize_parameter(self, pruning_settings, 'pruning_momentum', 0.9)
        initialize_parameter(self, pruning_settings, 'flops_regularization', 0.0)
        self.momentum_coeff = self.pruning_momentum
        self.use_momentum = self.pruning_momentum > 0.0

        initialize_parameter(self, pruning_settings, 'prune_per_iteration', 1)
        initialize_parameter(self, pruning_settings, 'start_pruning_after_n_iterations', 0)
        initialize_parameter(self, pruning_settings, 'prune_neurons_max', 0)
        initialize_parameter(self, pruning_settings, 'maximum_pruning_iterations', 0)
        initialize_parameter(self, pruning_settings, 'pruning_silent', False)
        initialize_parameter(self, pruning_settings, 'l2_normalization_per_layer', False)
        initialize_parameter(self, pruning_settings, 'fixed_criteria', False)
        initialize_parameter(self, pruning_settings, 'do_after_step', False)
        initialize_parameter(self, pruning_settings, 'set_moment_zero', False)
        initialize_parameter(self, pruning_settings, 'reset_gates', False)
        initialize_parameter(self, pruning_settings, 'starting_neuron', 0)
        initialize_parameter(self, pruning_settings, 'frequency', 30)
        initialize_parameter(self, pruning_settings, 'pruning_threshold', 100)
        initialize_parameter(self, pruning_settings, 'fixed_layer', -1)
        initialize_parameter(self, pruning_settings, 'combination_ID', 0)
        initialize_parameter(self, pruning_settings, 'seed', 0)
        initialize_parameter(self, pruning_settings, 'group_size', 1)
        initialize_parameter(self, pruning_settings, 'leave_at_least_one_group', True)



        initialize_parameter(self, pruning_settings, 'method', 0)
        initialize_parameter(self, pruning_settings, 'flops_regularization', 0.0)
        initialize_parameter(self, pruning_settings, 'params_regularization', 0.0)
        initialize_parameter(self, pruning_settings, 'latency_regularization', 0.0)
        initialize_parameter(self, pruning_settings, 'latency_look_up_table', "")

        initialize_parameter(self, pruning_settings, 'push_down', False)
        initialize_parameter(self, pruning_settings, 'push_down_weight_decay', 1e-3)

        print("\n\n Pruning regularizations:", {"FLOPS:":self.flops_regularization, "Params:":self.params_regularization , "Latency:":self.latency_regularization},"\n\n")

        self.latency_regularization = latency_regularization
        self.latency_look_up_table = latency_look_up_table

        # Hessian related parameters
        self.temp_hessian = [] # list to store Hessian
        self.hessian_first_time = True

        self.parameters = list()
        self.layers_group = list()

        self.pruning_parameters = for_pruning_parameters

        self.layers_group = [-1,] * len(self.pruning_parameters)

        ##get pruning parameters
        for layer in range(len(self.pruning_parameters)):
            # parameter_value = parameter["parameter"]
            self.parameters.append(self.pruning_parameters[layer]["compute_criteria_from"][0]["parameter"])


        #for previous versions
        for param in self.pruning_parameters:
            for layer_to_zero in param["set_to_zero"]:
                if "shift" not in layer_to_zero.keys():
                    layer_to_zero["shift"] = 0


        if self.fixed_layer == -1:
            ##prune all layers
            self.prune_layers = [True for parameter in self.parameters]
        else:
            ##prune only one layer
            self.prune_layers = [False, ]*len(self.parameters)
            self.prune_layers[self.fixed_layer] = True

        self.iterations_done = 0

        self.prune_network_criteria = list()
        self.prune_network_accomulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}

        self.pruning_gates = list()
        for layer in range(len(self.parameters)):
            self.prune_network_criteria.append(list())

            for key in self.prune_network_accomulate.keys():
                self.prune_network_accomulate[key].append(list())

            self.pruning_gates.append(np.ones(len(self.parameters[layer]),))
            layer_now_criteria = self.prune_network_criteria[-1]
            for unit in range(len(self.parameters[layer])):
                layer_now_criteria.append(0.0)

        # logging setup
        self.log_folder = log_folder
        self.folder_to_write_debug = self.log_folder + '/debug/'

        if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
            if not os.path.exists(self.folder_to_write_debug):
                os.makedirs(self.folder_to_write_debug)

        self.method_25_first_done = True

        if self.method == 40 or self.method == 50 or self.method == 25 or self.method == 26:
            self.oracle_dict = {"layer_pruning": -1, "initial_loss": 0.0, "loss_list": list(), "neuron": list(), "iterations": 0}
            self.method_25_first_done = False

        if (self.method == 25) or (self.method == 26):
            # with open("./pruning_core/plots/jasper/oracle.pickle", "rb") as f:
            with open("./pruning_core/oracle.pickle", "rb") as f:
                oracle_list = pickle.load(f)

            self.oracle_dict["loss_list"] = oracle_list

        self.needs_hessian = False
        if self.method in [10, 11]:
            self.needs_hessian = True

        # useful for storing data of the experiment
        self.data_logger = dict()
        self.data_logger["pruning_neurons"] = list()
        self.data_logger["pruning_accuracy"] = list()
        self.data_logger["pruning_loss"] = list()
        self.data_logger["method"] = self.method
        self.data_logger["prune_per_iteration"] = self.prune_per_iteration
        self.data_logger["combination_ID"] = list()
        self.data_logger["fixed_layer"] = self.fixed_layer
        self.data_logger["frequency"] = self.frequency
        self.data_logger["starting_neuron"] = self.starting_neuron
        self.data_logger["use_momentum"] = self.use_momentum

        self.data_logger["time_stamp"] = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        if hasattr(self, 'seed'):
            self.data_logger["seed"] = self.seed

        self.data_logger["filename"] = "%s/data_logger_seed_%d_%s.p"%(log_folder, self.data_logger["seed"], self.data_logger["time_stamp"])
        if self.method == 50:
            self.data_logger["filename"] = "%s/data_logger_seed_%d_neuron_%d_%s.p"%(log_folder, self.starting_neuron, self.data_logger["seed"], self.data_logger["time_stamp"])
        self.log_folder = log_folder

        # the rest of initializations
        self.pruned_neurons = self.starting_neuron

        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0.0

        self.loss_tracker_exp = ExpMeter()
        # stores results of the pruning, 0 - unsuccessful, 1 - successful
        self.res_pruning = 0

        self.iter_step = -1

        self.train_writer = None

        # self.set_moment_zero = True
        # self.set_moment_zero = False
        self.pruning_mask_from = ""

        self.needs_broadcast = False

        if LATENCY_COMPUTE:
            latency_file = "latency.json"
            if len(self.latency_look_up_table) > 0:
                latency_file = self.latency_look_up_table
            if not os.path.isfile(latency_file):
                print("Latency table cant be loaded, disabling latency reg")
                self.latency_look_up_table = list()
                self.latency_regularization = 0.0
            else:
                print("Loading latency look up table from: ", latency_file)
                with open(latency_file) as json_file:
                    measurement = json.load(json_file)
                EMB = np.arange(4)*256
                head = np.array([1,3,6,9,12])
                QK = np.array([1,16,32,48,64])
                V = np.array([1,16,32,48,64])
                MLP = (np.arange(25))*128
                data = np.zeros([4,5,5,5,25])
                for i in range(3):
                    for j in range(5):
                        for k1 in range(5):
                            for k2 in range(5):
                                for l in range(25):
                                    e = EMB[i+1]
                                    q_h = head[j]
                                    q = QK[k1]
                                    v = V[k2]
                                    if MLP[l]:
                                        h = MLP[l]
                                    else:
                                        h = 1
                                    data[i+1,j,k1,k2,l] = measurement['EMB_'+str(e)]['QK_'+str(q_h)+'_'+str(q)]['V_'+str(v)]['MLP_'+str(h)]
                self.latency_look_up_table = RegularGridInterpolator((EMB, head, QK, V, MLP), data)                
                original_latency = self.latency_look_up_table(np.array([768,12,64,64,3072]))*12
                self.current_latency = original_latency
                self.latency_target = original_latency*latency_target

        self.threshold_now = 0

        self.pruning_helper = None

        self.is_finished = False
        self.full_latency = -1

        self.overlap_score = 0.0

        self.total_prunable_channels = sum([len(a) for a in self.parameters])




    def init_pruning_helper(self, model, data, skip_pass=False):
        from .pruning_helper import pruning_helper, get_conv2d_sizes

        if not skip_pass:
            output_sizes = get_conv2d_sizes(model, data)
        else:
            output_sizes = None

        named_parameters = model.named_parameters()
        parameters = model.parameters()

        self.pruning_helper = pruning_helper(parameters=parameters, named_parameters=named_parameters, output_sizes=output_sizes)

        # import pdb; pdb.set_trace()


    def add_criteria(self):
        '''
        This method adds criteria to global list given batch stats.
        Works for Method 22 that uses BN parameters and their gradients
        '''

        if self.fixed_criteria:
            if self.pruning_iterations_done > self.start_pruning_after_n_iterations :
                return 0

        #imp = np.zeros((198,12))
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            nunits = self.parameters[layer].size(0)
            eps = 1e-8
            #print(layer)
            if self.method == 22:
                for w_ind, w in enumerate(self.pruning_parameters[layer]["compute_criteria_from"]):

                    if w["parameter"].grad is None:
                        NotImplementedError("parameter doesn't have gradient,", w["name"])
                        value = w["parameter"]*0.0
                    else:
                        value = w["parameter"] * w["parameter"].grad

                    if w["dim"] != 0:
                        raise NotImplementedError("Supports agregation only for the 0 dim in pruning criteria evaluation")
                    if w_ind==0:
                        criteria_for_layer = value.data.pow(2).view(nunits, -1).sum(dim=1)
                    else:
                        criteria_for_layer += value.data.pow(2).view(nunits, -1).sum(dim=1)


            if self.method == 23:
                # compute criteria from multiple inputs
                # in this case we will compute contribution as expectation of the loss change if channel is set to 0
                nunits = self.parameters[layer].size(0)
                Emb=False
                head=False
                qk=False
                for w_ind, w in enumerate(self.pruning_parameters[layer]["compute_criteria_from"]):

                    if w["parameter"].grad is None:
                        NotImplementedError("parameter doesn't have gradient,", w["parameter_name"])
                        value = w["parameter"]*0.0
                    else:
                        value = w["parameter"] * w["parameter"].grad

                    if 'EMB' in w["parameter_name"]:
                        Emb=True
                        #print(w["parameter_name"])
                    if not Emb:
                        if 'qkv.V' in w["parameter_name"] or 'qkv.Q' in w["parameter_name"]:
                            qk=True
                        if 'head_mask' in w["parameter_name"]:
                            head=True
                            value = w["parameter"]*0.0
                    #    print(w["parameter_name"])


                    nunits = w["parameter"].size(w["dim"])
                    #print((w_ind,nunits))
                    if w["dim"] == 0:
                        new_criteria = value.data.reshape(nunits, -1).sum(dim=1)
                    elif w["dim"] == 1:
                        if len(value.data.shape) == 2:
                            new_criteria = value.data.permute(1, 0).reshape(nunits, -1).sum(dim=1)
                        else:
                            new_criteria = value.data.permute(1, 0, 2).reshape(nunits, -1).sum(dim=1)
                    elif w["dim"] == 2:
                        new_criteria = value.data.permute(2, 0, 1).reshape(nunits, -1).sum(dim=1)

                    if w_ind == 0:
                        criteria_for_layer = new_criteria
                    else:
                        criteria_for_layer += new_criteria

                    # import pdb;pdb.set_trace()

                    if (~torch.isfinite(criteria_for_layer)).sum() > 0:
                        criteria_for_layer[~torch.isfinite(criteria_for_layer)] = 0.0
                        print("pruning_engine_general.py: NAN detected in add_criteria(), replacing with 0")
                        # pdb.set_trace()

                if Emb:
                    criteria_for_layer = criteria_for_layer/12.
                if head:
                    criteria_for_layer = criteria_for_layer/6.
                if qk:
                    criteria_for_layer = criteria_for_layer/2.
                
                if self.pruning_parameters[layer]["compute_criteria_from"][0]['fix']:
                    criteria_for_layer = torch.where(criteria_for_layer==0., criteria_for_layer, torch.ones_like(criteria_for_layer)*1e5)
                else:
                    criteria_for_layer = criteria_for_layer.pow(2)
                

            if self.method == 2:
                # compute criteria from multiple inputs
                # value = sum([w["parameter"] * w["parameter"].grad for w in self.pruning_parameters[layer]["compute_criteria_from"]])
                # value = sum([w["parameter"]**2 for w in self.pruning_parameters[layer]["compute_criteria_from"]])

                for w_ind, w in enumerate(self.pruning_parameters[layer]["compute_criteria_from"]):
                    if w_ind > 0:
                        continue
                        #not clear how to aggregate multiple criterias, we assume that the first one is the most important
                    if w["parameter"] is None:
                        NotImplementedError("parameter doesn't have gradient,", w["name"])
                        value = w["parameter"]*0.0
                    else:
                        value = w["parameter"]

                criteria_for_layer = value.data.pow(2).view(nunits, -1).sum(dim=1)

            if self.method == 40:
                # ORACLE on the fly that reevaluates itslef every pruning step
                criteria_for_layer = np.asarray(self.oracle_dict["loss_list"][layer]).copy()
                self.oracle_dict["loss_list"][layer] = list()

            if self.method == 25:
                # ORACLE from precomputed oracle
                # import pdb; pdb.set_trace()
                criteria_for_layer = np.asarray(self.oracle_dict["loss_list"][layer]).copy()

            if self.method == 26:
                # ORACLE from precomputed oracle
                # import pdb; pdb.set_trace()
                criteria_for_layer_oracle = np.asarray(self.oracle_dict["loss_list"][layer]).copy()

                for w_ind, w in enumerate(self.pruning_parameters[layer]["compute_criteria_from"]):
                    if w["parameter"].grad is None:
                        NotImplementedError("parameter doesn't have gradient,", w["name"])
                        value = w["parameter"] * 0.0
                    else:
                        value = w["parameter"] * w["parameter"].grad
                    if w_ind==0:
                        criteria_for_layer = value.data.view(nunits, -1).sum(dim=1)
                    else:
                        criteria_for_layer += value.data.view(nunits, -1).sum(dim=1)

                #methodd 26 is orracle + criteria
                # import pdb; pdb.set_trace()
                criteria_for_layer = criteria_for_layer.pow(2) + torch.from_numpy(criteria_for_layer_oracle**2).type(criteria_for_layer.type())

            if self.iterations_done == 0:
                self.prune_network_accomulate["by_layer"][layer] = criteria_for_layer
            else:
                self.prune_network_accomulate["by_layer"][layer] += criteria_for_layer
#            #print(criteria_for_layer.detach().cpu().numpy())
#            if head:
#                print(criteria_for_layer.detach().cpu().numpy())
#                imp = np.log10(criteria_for_layer.detach().cpu().numpy())
#                plt.figure()
#                plt.hist(imp, bins=30)
#                plt.title(w["parameter_name"] + ' Importance distribution')
#                plt.savefig('Importance distribution layer '+str(layer))
#                print(w["parameter_name"])
#        exit()
        self.iterations_done += 1
#        if self.iterations_done == 5:
#            exit()

    @staticmethod
    def group_criteria(list_criteria_per_layer, layers_group, group_size=1):
        '''
        Function combine criteria per neuron into groups of size group_size.
        Output is a list of groups organized by layers. Length of output is a number of layers.
        The criterion for the group is computed as an average of member's criteria.
        Input:
        list_criteria_per_layer - list of criteria per neuron organized per layer
        group_size - number of neurons per group
        layers_group - layers can form a group, e.g. residual connection, they will be pruned together

        Output:
        groups - groups organized per layer. Each group element is a tuple of 2: (index of neurons, criterion)
        groups_unique - groups organized per UNIQUE layers only. Each group element is a tuple of 2: (index of neurons, criterion)
        '''
        assert len(list_criteria_per_layer) == len(layers_group)
        groups = list()

        for layer_indx, layer_criteria in enumerate(list_criteria_per_layer):
            layer = layer_criteria

            if layer_indx == 0:
                group_size = 16
            else:
                if layer_indx%4 == 1:
                    group_size = 2
                elif layer_indx%4 == 2:
                    group_size = 8
                elif layer_indx%4 == 3:
                    group_size = 8
                elif layer_indx%4 == 0:
                    group_size = 16

            if layers_group[layer_indx] != -1:
                
                #if layer/parameter is a part of the group
                #then we aggregate importance across the group
                #this procedure is repeated for each layer/parameter in the group

                all_criteria = np.asarray([lc for li, lc in enumerate(list_criteria_per_layer) if layers_group[layer_indx]==layers_group[li]])

                all_criteria = all_criteria.sum(0)
                layer = all_criteria

            groups_in_layer = list()
            indeces = np.argsort(layer)
            for group_id in range(int(np.ceil(len(layer)/group_size))):
                current_group = slice(group_id*group_size, min((group_id+1)*group_size, len(layer)))
                values = [layer[ind] for ind in indeces[current_group]]
                group = [indeces[current_group], sum(values)]

                groups_in_layer.append(group)
            groups.append(groups_in_layer)

        if all( [l==-1 for l in layers_group]):
            #group with index -1 means no group
            unique_groups = groups
        else:
            unique_groups = list()
            groups_exists = list()
            for gi, g in enumerate(groups):
                if (layers_group[gi] == -1) or (layers_group[gi] not in groups_exists):
                    unique_groups.append(g)
                    groups_exists.append(layers_group[gi])

        # if torch.distributed.get_rank() == 0:
        #     import pdb;pdb.set_trace()

        return groups, unique_groups

    def compute_saliency(self):
        '''
        Method performs pruning based on precomputed criteria values. Needs to run after add_criteria()
        Apply latency_look_up_table to unfixed layers
        '''
        import pickle
        with open("/workspace/alex/NViT/nvit/pre_compute_saliency.pkl", 'wb') as f:
            pickle.dump(self, f)
        exit()
        def write_to_debug(what_write_name, what_write_value):
            # Aux function to store information in the text file
            with open(self.log_debug, 'a') as f:
                f.write("{} {}\n".format(what_write_name,what_write_value))

        def nothing(what_write_name, what_write_value):
            pass

        #store the mask for future needs
        old_mask = copy.deepcopy(self.pruning_gates)

        if self.method == 50:
            write_to_debug = nothing

        if not (not torch.distributed.is_initialized() or torch.distributed.get_rank()==0):
            write_to_debug = nothing

        # compute loss since the last pruning and decide if to prune:
        if self.util_loss_tracker_num > 0:
            # validation_error = self.util_loss_tracker / self.util_loss_tracker_num
            validation_loss = self.util_loss_tracker / self.util_loss_tracker_num
            # validation_error_long = validation_error
            acc = self.util_acc_tracker / self.util_loss_tracker_num
        else:
            print("compute loss and run self.util_add_loss(loss.item()) before running this")
            validation_error = 0.0
            acc = 0.0
            validation_loss = 0.0

        self.util_training_loss = validation_loss
        self.util_training_acc = acc

        # reset training loss tracker
        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0


        if (validation_loss > self.pruning_threshold) and (self.pruning_threshold != -1.0):
            ## if error is big then skip pruning
            print("skipping pruning because current loss is: ", validation_loss, "while limit is set to", self.pruning_threshold)
            if self.method != 4:
                self.res_pruning = -1
                return -1

        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        self.full_list_of_criteria = list()

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if self.iterations_done > 0:
                # momentum turned to be useless and even reduces performance
                contribution = self.prune_network_accomulate["by_layer"][layer] / self.iterations_done
                # import pdb; pdb.set_trace()
                if len(self.prune_network_accomulate["averaged"][layer])==0 or not self.use_momentum or (self.method in [4, 40, 50, 25]):
                    self.prune_network_accomulate["averaged"][layer] = contribution
                else:
                    # use momentum to accumulate criteria over several pruning iterations:
                    self.prune_network_accomulate["averaged"][layer] = self.momentum_coeff*self.prune_network_accomulate["averaged"][layer]+(1.0- self.momentum_coeff)*contribution

                current_layer = self.prune_network_accomulate["averaged"][layer]
                if not (self.method in [1, 4, 40, 15, 50, 25]):
                    current_layer = current_layer.cpu().numpy()

                if self.l2_normalization_per_layer:
                    eps = 1e-8
                    current_layer = current_layer / (np.linalg.norm(current_layer) + eps)

                self.prune_network_accomulate["averaged_cpu"][layer] = current_layer
            else:
                print("First do some add_criteria iterations")
                exit()


            for unit in range(len(self.parameters[layer])):
                criterion_now = current_layer[unit]

                # make sure that pruned neurons have 0 criteria
                if not self.push_down:
                    self.prune_network_criteria[layer][unit] =  criterion_now * self.pruning_gates[layer][unit]
                else:
                    self.prune_network_criteria[layer][unit] =  criterion_now

                if self.method == 50:
                    self.prune_network_criteria[layer][unit] =  criterion_now

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.neuron_units = neuron_units
        self.all_neuron_units = all_neuron_units


        if not self.fixed_criteria:
            self.iterations_done = 0

        prune_network_criteria_updated = self.prune_network_criteria
        
        # Compute current model statistic
        model_dim = list()
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue
            if layer == 0:
                model_dim.append(np.nonzero(self.pruning_gates[layer])[0].size)
            else:
                if layer%4 == 1:
                    head = np.nonzero(self.pruning_gates[layer])[0].size
                elif layer%4 == 2:
                    qk = np.nonzero(self.pruning_gates[layer])[0].size
                elif layer%4 == 3:
                    v = np.nonzero(self.pruning_gates[layer])[0].size
                elif layer%4 == 0:
                    mlp = np.nonzero(self.pruning_gates[layer])[0].size
                    model_dim.append({'head':head,'QK':qk,'V':v,'MLP':mlp})

        # Compute latency and adjust importance
        if self.latency_regularization:
            for layer, if_prune in enumerate(self.prune_layers):
                if not if_prune:
                    continue
                if layer==0 and not self.pruning_parameters[layer]["compute_criteria_from"][0]['fix']:
                    emb = model_dim[0]
                    latency_improve = 0.
                    for blk in range(12):
                        qk_head = model_dim[blk+1]['head']
                        qk = model_dim[blk+1]['QK']
                        v = model_dim[blk+1]['V']
                        mlp = model_dim[blk+1]['MLP']
                        latency_improve += self.compute_latency(emb,qk_head,qk,v,mlp)-self.compute_latency(emb-1,qk_head,qk,v,mlp)
                    pc = np.array(prune_network_criteria_updated[layer])
                    pc -= self.latency_regularization*latency_improve
                    prune_network_criteria_updated[layer] = pc.tolist()
                elif not self.pruning_parameters[layer]["compute_criteria_from"][0]['fix']:
                    emb = model_dim[0]
                    qk_head = model_dim[(layer-1)//4+1]['head']
                    qk = model_dim[(layer-1)//4+1]['QK']
                    v = model_dim[(layer-1)//4+1]['V']
                    mlp = model_dim[(layer-1)//4+1]['MLP']
                    #print(emb,qk,v,mlp)
                    latency_improve = 0.
                    if layer%4 == 1 and qk_head>2:
                        latency_improve = self.compute_latency(emb,qk_head,qk,v,mlp)-self.compute_latency(emb,qk_head-1,qk,v,mlp)
                    elif layer%4 == 2 and qk>8:
                        latency_improve = self.compute_latency(emb,qk_head,qk,v,mlp)-self.compute_latency(emb,qk_head,qk-1,v,mlp)
                    elif layer%4 == 3 and v>8:
                        latency_improve = self.compute_latency(emb,qk_head,qk,v,mlp)-self.compute_latency(emb,qk_head,qk,v-1,mlp)
                    elif layer%4 == 0 and mlp>16:
                        latency_improve = self.compute_latency(emb,qk_head,qk,v,mlp)-self.compute_latency(emb,qk_head,qk,v,mlp-1)
                    
                    pc = np.array(prune_network_criteria_updated[layer])#+1
                    pc -= self.latency_regularization*latency_improve
                    prune_network_criteria_updated[layer] = pc.tolist()
                    

        # create groups per layer
        groups, unique_groups = self.group_criteria(prune_network_criteria_updated, layers_group = self.layers_group, group_size=self.group_size)

        # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in unique_groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruning_iterations_done * self.prune_per_iteration)//self.group_size - 1
        if self.push_down:
            removed_gates = sum([(a==0.0).sum() for a in self.pruning_gates])
            prune_additionally = self.prune_per_iteration
            prune_neurons_now = (removed_gates + prune_additionally) // self.group_size - 1

        if self.prune_neurons_max != -1:
            prune_neurons_now = max(0,min(len(all_criteria)-1, min(prune_neurons_now, self.prune_neurons_max//self.group_size - 1)))

        if self.push_down:
            self.reset_gates_to_1()

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]
        # import pdb; pdb.set_trace()
        if np.isnan(threshold_now):
            print("skipping")
            return 0


        self.pruning_iterations_done += 1

        self.log_debug = self.folder_to_write_debug + 'debugOutput_pruning_%08d' % (
            self.pruning_iterations_done) + '.txt'
        write_to_debug("method", self.method)
        write_to_debug("pruned_neurons", self.pruned_neurons)
        write_to_debug("pruning_iterations_done", self.pruning_iterations_done)
        write_to_debug("neuron_units", neuron_units)
        write_to_debug("all_neuron_units", all_neuron_units)
        write_to_debug("threshold_now", threshold_now)
        write_to_debug("groups_total", sum([len(layer) for layer in groups]))
        write_to_debug("uniquegroups_total", sum([len(layer) for layer in unique_groups]))

        if self.pruning_iterations_done < self.start_pruning_after_n_iterations:
            self.res_pruning = -1
            return -1


            if self.prune_per_iteration == 0:
                continue

            total_groups_in_layer = len(groups[layer])
            zeroed_groups = 0

            for group in groups[layer]:
                if group[1] <= threshold_now:

                    #add skip if all groups are set to zero in the current layer:
                    if (zeroed_groups >= total_groups_in_layer-1) and self.leave_at_least_one_group:
                        print("PRUNING: skipping the group because others are zero")
                        continue

                    zeroed_groups += 1
                    for unit in group[0]:
                        # do actual pruning
                        if self.leave_at_least_one_group and (self.pruning_gates[layer].sum()<=1):
                            print("PRUNING: skipping setting the last neuron to zero")
                            continue

                        self.pruning_gates[layer][unit] *= 0.0
                        

                        if not self.push_down:
                            for param in self.pruning_parameters[layer]["set_to_zero"]:
                                
                                if check_allow_trim(param):
                                    in_the_range = unit + param["shift"] < param["parameter"].data.shape[param["dim"]]
                                    if (not in_the_range) or not(unit + param["shift"] >= 0):
                                        continue

                                if param["dim"] == 0:
                                    param["parameter"].data[unit + param["shift"]] *= 0.0
                                elif param["dim"] == 1:
                                    param["parameter"].data[:, unit + param["shift"]] *= 0.0
                                elif param["dim"] == 2:
                                    param["parameter"].data[:, :, unit + param["shift"]] *= 0.0

            write_to_debug("pruned_perc:", [np.nonzero(1.0-self.pruning_gates[layer])[0].size, len(self.pruning_gates[layer])])

        # count number of neurons
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            model_dim = np.zeros((1,49))
            latency = 0.
            for layer, if_prune in enumerate(self.prune_layers):
                if not if_prune:
                    continue
                if layer == 0:
                    model_dim[0,0] = np.nonzero(self.pruning_gates[layer])[0].size
                    self.train_writer.add_scalar('dimension/EMB', model_dim[0,0], self.pruning_iterations_done)
                else:
                    if layer%4 == 1:
                        qk_head = np.nonzero(self.pruning_gates[layer])[0].size
                        self.train_writer.add_scalar('dimension/Layer_%d_head' % (layer//4+1), qk_head, self.pruning_iterations_done)
                        model_dim[0,layer//4+1] = qk_head
                    elif layer%4 == 2:
                        qk = np.nonzero(self.pruning_gates[layer])[0].size
                        self.train_writer.add_scalar('dimension/Layer_%d_QK' % (layer//4+1), qk, self.pruning_iterations_done)
                        model_dim[0,layer//4+13] = qk
                    elif layer%4 == 3:
                        v = np.nonzero(self.pruning_gates[layer])[0].size
                        self.train_writer.add_scalar('dimension/Layer_%d_V' % (layer//4+1), v, self.pruning_iterations_done)
                        model_dim[0,layer//4+25] = v
                    elif layer%4 == 0:
                        mlp = np.nonzero(self.pruning_gates[layer])[0].size
                        model_dim[0,layer//4+36] = mlp
                        latency += self.compute_latency(emb,qk_head,qk,v,mlp)
                        self.train_writer.add_scalar('dimension/Layer_%d_MLP' % (layer//4), mlp, self.pruning_iterations_done)
            
            self.current_latency = latency
            #print(latency)
            self.train_writer.add_scalar('dimension/Estimated_Latency', latency, self.pruning_iterations_done)
            log_dim = self.folder_to_write_debug + 'debugOutput_dimension_%08d' % (self.pruning_iterations_done) + '.txt'
            np.savetxt(log_dim, model_dim, fmt='%d', delimiter=',')

        all_neuron_units, neuron_units = self._count_number_of_neurons()

        self.pruned_neurons = all_neuron_units-neuron_units

        if self.method == 25:
            self.method_25_first_done = True

        self.threshold_now = threshold_now
        try:
            self.min_criteria_value = (all_criteria[all_criteria > 0.0]).min()
            self.max_criteria_value = (all_criteria[all_criteria > 0.0]).max()
            self.median_criteria_value = np.median(all_criteria[all_criteria > 0.0])

            self.min_max_crit_stats =list()
            for layer_id, layer in enumerate(unique_groups):
                criterias_group = np.asarray([group[1] for group in layer])
                min_c = criterias_group[criterias_group>0.0].min()
                max_c = criterias_group[criterias_group>0.0].max()
                mean_c = criterias_group[criterias_group>0.0].mean()
                self.min_max_crit_stats.append({"min": min_c, "max": max_c, "mean_c": mean_c})

        except:
            self.min_criteria_value = 0.0
            self.max_criteria_value = 0.0
            self.median_criteria_value = 0.0

        #get overlap
        self.overlap_score = self.compute_mask_overlap(old_mask, self.pruning_gates)

        # set result to successful
        self.res_pruning = 1

        # self.pruning_helper = None
        #output to a single file:
        if not self.pruning_silent:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                with open(self.log_folder + "pruner_all_info.txt", 'a') as f:
                    f.write(f"Pruning_iterations_done: {self.pruning_iterations_done}\n")
                    for layer, if_prune in enumerate(self.prune_layers):
                        if not if_prune:
                            continue
                        f.write(f"Layer:\t{layer}\ttotal: {len(self.pruning_gates[layer])}\t active: {int(self.pruning_gates[layer].sum())}\t pruned {int(len(self.pruning_gates[layer])-self.pruning_gates[layer].sum())}\n")
                    f.write("\n")


    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        #PAVLO: Why not to count from pruning gates? probably because of groups? hat we can combine?
        all_neuron_units = 0
        neuron_units = 0
        checked_groups = list()
        for layer, if_prune in enumerate(self.prune_layers):

            if not if_prune:
                continue

            if (self.layers_group[layer] != -1):
                if (self.layers_group[layer] not in checked_groups):
                    checked_groups.append(self.layers_group[layer])
                else:
                    continue

            all_neuron_units += len( self.parameters[layer] )

            if self.pruning_parameters[layer]["compute_criteria_from"][0]["dim"] == 0:
                ndim_c = self.parameters[layer].shape[0]
                non_zero_parameters = abs(self.parameters[layer].reshape(ndim_c, -1)).sum(dim=1)
                non_zero_parameters = (non_zero_parameters > 0).sum()
            elif self.pruning_parameters[layer]["compute_criteria_from"][0]["dim"] == 1:
                ndim_c = self.parameters[layer].shape[1]
                if len(self.parameters[layer].shape)==2:
                    non_zero_parameters = abs(self.parameters[layer].permute(1, 0).reshape(ndim_c, -1)).sum(dim=1)
                else:
                    non_zero_parameters = abs(self.parameters[layer].permute(1, 0, 2).reshape(ndim_c, -1)).sum(dim=1)
                non_zero_parameters = (non_zero_parameters > 0).sum()
            elif self.pruning_parameters[layer]["compute_criteria_from"][0]["dim"] == 2:
                ndim_c = self.parameters[layer].shape[2]
                non_zero_parameters = abs(self.parameters[layer].permute(2, 0, 1).reshape(ndim_c, -1)).sum(dim=1)
                non_zero_parameters = (non_zero_parameters > 0).sum()

            neuron_units += non_zero_parameters.item()

        return all_neuron_units, neuron_units

    def reset_gates_to_1(self):
        '''
        Method sets gates and parameters to 1 such that model can recover from pruning wrong neurons
        :return:
        '''
        # print("PRUNING: reseting gates to 1")
        for layer, if_prune in enumerate(self.pruning_gates):
            for unit in range(len(self.pruning_gates[layer])):
                self.pruning_gates[layer][unit] = 1.0



    def enforce_pruning(self):
        '''
        Method sets parameters ang gates to 0 for pruned neurons.
        Helpful if optimizer will change weights from being zero (due to regularization etc.)
        '''
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            zeroed_el = np.nonzero(1.0 - self.pruning_gates[layer])[0]
            if len(zeroed_el) == 0:
                continue

            if (not self.push_down) or (self.is_finished):
                for param in self.pruning_parameters[layer]["set_to_zero"]:
                    set_to_zero = zeroed_el + param["shift"]
                    if check_allow_trim(param):
                        in_the_range = zeroed_el + param["shift"] < param["parameter"].data.shape[param["dim"]]
                        in_the_range = in_the_range*(zeroed_el + param["shift"] >= 0)
                        set_to_zero = set_to_zero[in_the_range]

                    # print("allow_trim", in_the_range, unit + param["shift"], param["parameter"].data.shape[param["dim"]], self.pruning_gates[layer].shape)

                    if param["dim"] == 0:
                        param["parameter"].data[set_to_zero] *= 0.0
                    elif param["dim"] == 1:
                        param["parameter"].data[:, set_to_zero] *= 0.0
                    elif param["dim"] == 2:
                        param["parameter"].data[:, :, set_to_zero] *= 0.0


    def report_loss_neuron(self, training_loss, training_acc, train_writer = None, neurons_left = 0):
        '''
        method to store stistics during pruning to the log file
        :param training_loss:
        :param training_acc:
        :param train_writer:
        :param neurons_left:
        :return:
        void
        '''
        if train_writer is not None:
            train_writer.add_scalar('loss_neuron', training_loss, self.all_neuron_units-self.neuron_units)

        self.data_logger["pruning_neurons"].append(self.all_neuron_units-self.neuron_units)
        self.data_logger["pruning_loss"].append(training_loss)
        self.data_logger["pruning_accuracy"].append(training_acc)

        self.write_log_file()

    def write_log_file(self):
        with open(self.data_logger["filename"], "wb") as f:
            pickle.dump(self.data_logger, f)

    def load_mask(self):
        '''Method loads precomputed criteria for pruning
        :return:
        '''
        if not len(self.pruning_mask_from)>0:
            print("pruning_engine.load_mask(): did not find mask file, will load nothing")
        else:
            if not os.path.isfile(self.pruning_mask_from):
                print("pruning_engine.load_mask(): file doesn't exist", self.pruning_mask_from)
                print("pruning_engine.load_mask(): check it, exit,", self.pruning_mask_from)
                exit()

            with open(self.pruning_mask_from, 'rb') as f:
                self.loaded_mask_criteria = pickle.load(f)

            print("pruning_engine.load_mask(): loaded criteria from", self.pruning_mask_from)



    def report_to_tensorboard(self, train_writer, processed_batches):
        '''
        Log data with tensorboard
        '''
        if train_writer is None:
            return 0

        gradient_norm_final_before = self.gradient_norm_final
        train_writer.add_scalar('Neurons_left', self.neuron_units, processed_batches)
        train_writer.add_scalar('criteria/Criteria_min', self.min_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('criteria/Criteria_max', self.max_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('criteria/Criteria_median', self.median_criteria_value, self.pruning_iterations_done)
        # train_writer.add_scalar('Gradient_norm_before', gradient_norm_final_before, self.pruning_iterations_done)
        train_writer.add_scalar('criteria/Pruning_threshold', self.threshold_now, self.pruning_iterations_done)
        train_writer.add_scalar('pruning/overlap_score', self.overlap_score, self.pruning_iterations_done)

        train_writer.add_scalar('pruning/latency_batch', self.full_latency, self.pruning_iterations_done)

    def util_add_loss(self, training_loss_current, training_acc):
        # keeps track of current loss
        self.util_loss_tracker += training_loss_current
        self.util_acc_tracker  += training_acc
        self.util_loss_tracker_num += 1
        self.loss_tracker_exp.update(training_loss_current)
        # self.acc_tracker_exp.update(training_acc)

    def do_step(self, loss=None, optimizer=None, neurons_left=0, training_acc=0.0):
        '''
        do one step of pruning,
        1) Add importance estimate
        2) checks if loss is above threshold
        3) performs one step of pruning if needed
        '''
        DO_ONCE_GPU = (not torch.distributed.is_initialized() or torch.distributed.get_rank()==0)

        self.iter_step += 1
        niter = self.iter_step

        # # sets pruned weights to zero
        self.enforce_pruning()

        if self.iter_step == 0:
            self.update_flops_stats()

        # print(loss)
        #check if we have valid gradients:
        # if torch.isnan(self.parameters[0].grad.sum()):
        #     print("PRUNING: skip because grad is NaN")
        #     return -1

        # stop if latency target is achieved
        if self.current_latency <= self.latency_target:
            self.res_pruning = -1
            self.is_finished = True
            return -2

        # stop if pruned maximum amount
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # exit if we pruned enough
            self.res_pruning = -1
            self.is_finished = True
            return -1


        # compute criteria for given batch
        self.add_criteria()

        # small script to keep track of training loss since the last pruning
        self.util_add_loss(loss, training_acc)

        if ((niter-1) % self.frequency == 0) and (niter != 0) and (self.res_pruning==1):
            self.report_loss_neuron(self.util_training_loss, training_acc=self.util_training_acc, train_writer=self.train_writer, neurons_left=neurons_left)

        if niter % self.frequency == 0 and niter != 0:
            # do actual pruning, output: 1 - good, 0 - no pruning

            self.compute_saliency()

            self.set_momentum_zero_sgd(optimizer=optimizer)

            training_loss = self.util_training_loss
            if self.res_pruning == 1:
                print("PRUNING: Units", self.neuron_units, "/", self.all_neuron_units, "loss", training_loss, "Zeroed",
                      self.pruned_neurons,
                      "criteria min:{}/max:{:2.7f}".format(self.min_criteria_value, self.max_criteria_value))
                if DO_ONCE_GPU:
                    # import pdb; pdb.set_trace()
                    if self.pruning_helper is not None:
                        self.pruning_helper.step_after()
                        self.pruning_helper.report_to_tensorboard(self.train_writer, global_iteration = niter)

        if ((niter-1) % self.frequency == 0):

            self.report_to_tensorboard(self.train_writer, niter)
            #and ((niter-1) != 0):
            #step_after() will check if some weights are getting zero gradient and will be set to 0
            #we prune gates, but assosiated conv weights are not pruned, this function will set them to 0
            #it will compute number of neurons and flops left
            self.update_flops_stats()
            if self.do_after_step:
                self.group_wd_optimizer.step_after()

                #TODO: set gates back to 1 such that network can recover lost accuracy
                #self.reset_gates_next_step
                #for it we need to check if this is the last iteration



                if DO_ONCE_GPU:
                    if self.tensorboard:
                        neurons_left = int(self.group_wd_optimizer.get_number_neurons())
                        flops = int(self.group_wd_optimizer.get_number_flops(print_output=False))

                        if self.train_writer is not None:
                            self.train_writer.add_scalar('neurons_optimizer_left', neurons_left, self.iter_step)
                            self.train_writer.add_scalar('neurons_optimizer_flops_left', flops, self.iter_step)

            if self.reset_gates:
                #will set gates to 1 so that model can continue training
                self.reset_gates_to_1()

        ##set gradients to zero for gates:
        self.zero_grad()

    def zero_grad(self):
        return 0
        # for now no gates

        # make sure that all gradients are zero for the next step
        # self.pruning_parameters[layer]["compute_criteria_from"][0]
        for layer in self.pruning_parameters:
            params = layer["compute_criteria_from"]
            for p in params:
                if p["parameter"].grad is not None:
                    p["parameter"].grad.detach_()
                    p["parameter"].grad.zero_()

    def prestep(self, opt):
        #method calls internal functions to be executed before the main pruning step
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            return -1
        #do only during active pruning

        self.group_wd_optimizer.step_ext(param_groups=opt.param_groups, state=opt.state,
                                                   group_lasso_weight=self.group_lasso_weight)

        # self.group_wd_optimizer.step()


    def set_momentum_zero_sgd(self, optimizer=None):
        '''
        Method sets momentum buffer to zero for pruned neurons. Supports SGD only.
        :return:
        void
        '''
        for layer in range(len(self.pruning_gates)):
            if not self.prune_layers[layer]:
                continue
            for unit in range(len(self.pruning_gates[layer])):
                if not self.pruning_gates[layer][unit]:
                    continue
                if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0

        if (self.maximum_pruning_iterations==self.pruning_iterations_done) and self.set_moment_zero:
            print("Setting momentums to zero after pruning is finished")
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = optimizer.state[p]
                    if 'momentum_buffer' in param_state:
                        del param_state['momentum_buffer']
            import collections
            optimizer.state = collections.defaultdict(dict)

    def connect_tensorboard(self, tensorboard):
        '''
        Function connects tensorboard to pruning engine
        '''
        self.tensorboard = True
        self.train_writer = tensorboard

    def update_flops_stats(self):
        '''
        Function updates flops for potential regularization
        :param stats: a list of flops per parameter
        :return:
        '''
        return 0
        if self.latency_regularization <= 0.0:
            return 0

        self.per_layer_flops = list()
        self.per_layer_params = list()
        self.conv_params = list()
        self.latency_improve = list()

        # getting full latency:

        #getting latency change per layer
        full_model_latency = sum([a["full"] for a in self.latency_look_up_table]) / float(len(self.latency_look_up_table))

        if 1:
            current_full_model_latency = full_model_latency
            #get speed up from the pruning gates
            for layer_id, gate in enumerate(self.pruning_gates):
                # imporvement is
                latency_for_layer = self.get_gated_latency(layer_id, reduce_by=0)
                if latency_for_layer > 0:
                    latency_change = self.latency_look_up_table[layer_id]["full"] - latency_for_layer
                    # latency_improvement = self.latency_look_up_table[layer_id]["full"] - latency_for_layer
                    # pdb.set_trace()
                    current_full_model_latency -= latency_change

            # print(f"---Pruning: full model {full_model_latency:5.5f}\tcurrent {current_full_model_latency:5.5f}")

        if 1:
            group_size = self.group_size
            #latency for a pruned model taking into account the gate
            #better create a list of active neurons
            # active_neurons_per_layer = [a.sum() for a in self.pruning_gates]
            # size_neurons_per_layer = [a.size for a in self.pruning_gates]
            full_latency = full_model_latency
            ADD_ZERO_LATENCY = True
            if self.latency_regularization > 0.0:
                for gate_in in range(len(self.pruning_gates)):
                    if gate_in>=len(self.latency_look_up_table):
                        latency_change = 0
                    else:
                        active_channels = sum(self.pruning_gates[gate_in])

                        latency_current = self.get_gated_latency(gate_in, reduce_by=0)
                        reduced_latency_if_pruned = self.get_gated_latency(gate_in, reduce_by=group_size)
                        if ADD_ZERO_LATENCY:
                            lat_when_is_zero = self.latency_look_up_table[gate_in][0]

                        latency_change = latency_current - reduced_latency_if_pruned
                        # pdb.set_trace()
                        if ADD_ZERO_LATENCY:
                            # lat_when_is_zero_change = full_latency - lat_when_is_zero
                            lat_when_is_zero_change = self.latency_look_up_table[gate_in]["full"] - lat_when_is_zero
                            lat_when_is_zero_change = max(0.0, lat_when_is_zero_change)

                            # print(gate_in, ":", "latency loss:", latency_change, lat_when_is_zero_change, int(sum(self.pruning_gates[gate_in])), "/", len(self.pruning_gates[gate_in]))
                            # if store_num > 0:
                            if active_channels > 0:
                                latency_change = latency_change + 0.1*lat_when_is_zero_change/active_channels*group_size
                            else:
                                latency_change = 0.0

                        else:
                            pass
                            # print(gate_in, ":", "latency loss:", latency_change, len(self.pruning_gates[gate_in]), int(sum(self.pruning_gates[gate_in])), "/", len(self.pruning_gates[gate_in]))

                    latency_change = max(latency_change, 0.0)
                    self.latency_improve.append(latency_change)

            self.full_latency = current_full_model_latency
            # pdb.set_trace()
            # print("full_latency:", full_latency)

    def get_gated_latency(self, layer_id, reduce_by=0):
        if layer_id<len(self.latency_look_up_table):
            all_keys = self.latency_look_up_table[layer_id].keys()
            gate = self.pruning_gates[layer_id]
            active_channels = int(gate.sum())-reduce_by
            # find the closes representative
            all_keys = [a for a in all_keys if a!="full"]
            # pdb.set_trace()
            indx = np.argmin(abs(np.asarray(all_keys) - active_channels))
            latency = self.latency_look_up_table[layer_id][all_keys[indx]]
        else:
            latency = 0.0
        return latency


    def apply_stats_regularization(self, criteria, stats, mu=0.1, latency = False):
        '''
        Function applieregularisation to computed importance per layer
        :param groups: a list of groups organized per layer
        :param mu: regularization coefficient
        :return:
        '''
        # some_stats = list()
        #we assume stats are positive
        #TODO: try min-max normalization of stats instead of max only

        if len(stats) < 1:
            return -1

        updated_criteria = list()

        #convert importance to ratio to total importance
        #convert stat to total stat

        # total_importance = 0.0
        total_stat = 0.0
        total_stat_max = 0.0
        total_stat_min = 100000.0

        for layer_id, layer in enumerate(criteria):
            # total_importance += sum(layer)
            total_stat += stats[layer_id]

            if total_stat_max < stats[layer_id]:
                total_stat_max = stats[layer_id]
            if total_stat_min > stats[layer_id]:
                total_stat_min = stats[layer_id]

        if latency:
            total_stat_max = self.full_latency

        for layer_id, layer in enumerate(criteria):
            #add constrain as a regularization factor
            if 1 or not latency:
                multiplier = (1.0 - mu * (stats[layer_id] / total_stat_max))
                multiplier = min(multiplier, 1.0)
                multiplier = max(multiplier, 0.01)
                updated_criteria_layer = np.array(layer) * multiplier
                # print("multiplier: ", layer_id, "\t", multiplier)
            else:
                # updated_criteria_layer = np.array(layer) - 0.01*stats[layer_id]
                updated_criteria_layer = np.array(layer) - (1-mu)*stats[layer_id]
            updated_criteria.append(updated_criteria_layer)

        if latency:
            #print all latency penalization information
            if not self.pruning_silent:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    with open(self.log_folder + "pruner_latency_penalty.txt", 'a') as f:
                        f.write(f"Pruning_iterations_done: {self.pruning_iterations_done}\n")
                        for layer, if_prune in enumerate(self.prune_layers):
                            if not if_prune:
                                continue
                            multiplier = (1.0 - mu * (stats[layer] / total_stat_max))
                            multiplier = min(multiplier, 1.0)
                            multiplier = max(multiplier, 0.01)
                            if 0:
                                f.write(
                                    f"Layer:\t{layer}\tactive/total: {int(sum(self.pruning_gates[layer]))}/{len(self.pruning_gates[layer])}\t multiplier: {multiplier}\n")
                            else:
                                f.write(
                                    f"Layer:\t{layer}\tactive/total: {int(sum(self.pruning_gates[layer]))}/{len(self.pruning_gates[layer])}\t multiplier: {multiplier}\t")

                                if hasattr(self,"min_max_crit_stats"):
                                    if layer<len(self.min_max_crit_stats):
                                        f.write("{}\n".format(self.min_max_crit_stats[layer]))

                        f.write("\n")

        return updated_criteria

    def plot_histograms(self):
        #plots histogram for research study
        #group or gate?
        #criteria or not?
        # writer.add_histogram('hist', array, iteration)
        get_data = lambda x: x.data.cpu().numpy()
        get_data = lambda x: x.data.cpu().numpy()

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            self.train_writer.add_histogram('gates/layer_%02d'%layer, self.pruning_gates[layer], self.iter_step)
            self.train_writer.add_histogram('parameters/layer_%02d'%layer, get_data(self.parameters[layer]), self.iter_step)
            self.train_writer.add_histogram('parameters_norm/layer_%02d'%layer, np.linalg.norm(get_data(self.parameters[layer]).reshape((len(self.parameters[layer]),-1)), axis=1), self.iter_step)
            if len(self.prune_network_accomulate["averaged_cpu"][layer])>0:
                self.train_writer.add_histogram('criteria_ave/layer_%02d'%layer, np.asarray(self.prune_network_accomulate["averaged_cpu"][layer]), self.iter_step)
                self.train_writer.add_histogram('criteria_cur/layer_%02d'%layer, get_data(self.prune_network_accomulate["by_layer"][layer]), self.iter_step)


        #global
        get_data_global = lambda x: np.concatenate([a.data.cpu().numpy().reshape(-1) for a in x])
        get_data_global_norm = lambda x: np.concatenate([np.linalg.norm(a.data.cpu().numpy().reshape(len(a),-1), axis=1) for a in x])
        get_data_global_cpu = lambda x: np.concatenate([a.reshape(-1) for a in x])
        self.train_writer.add_histogram('global/parameters', get_data_global(self.parameters), self.iter_step)
        self.train_writer.add_histogram('global/parameters_norm', get_data_global_norm(self.parameters), self.iter_step)
        self.train_writer.add_histogram('global/gates', get_data_global_cpu(self.pruning_gates), self.iter_step)
        if len(self.prune_network_accomulate["averaged_cpu"][layer]) > 0:
            self.train_writer.add_histogram('global/criteria_ave', get_data_global(self.prune_network_accomulate["averaged"]), self.iter_step)
            self.train_writer.add_histogram('global/criteria_cur', get_data_global(self.prune_network_accomulate["by_layer"]), self.iter_step)


    def set_weights_oracle_pruning(self):
        '''
        sets gates/weights to zero to evaluate pruning
        will reuse weights for pruning
        only for oracle pruning
        '''

        for layer,if_prune in enumerate(self.prune_layers_oracle):
            if not if_prune:
                continue

            if self.method == 40:
                self.parameters[layer].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())
                # self.pruning_parameters[layer]["set_to_zero"][0]["parameter"].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())

            for unit in range(len(self.pruning_gates[layer])):
                if self.method == 40:
                    self.pruning_gates[layer][unit] = 1.0

                    if unit == self.oracle_unit:
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0

        return 1

    def reset_oracle_pruning(self):
        '''
        Method restores weights to original after masking for Oracle pruning
        :return:
        '''
        for layer, if_prune in enumerate(self.prune_layers_oracle):
            if not if_prune:
                continue

            if self.method == 40 or self.method == 50:
                self.parameters[layer].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())

            for unit in range(len( self.pruning_gates[layer])):
                if self.method == 40 or self.method == 50:
                    self.pruning_gates[layer][unit] = 1.0


    def run_full_oracle(self, model, data, target, criterion):
        '''
        Runs oracle on all data by setting to 0 every neuron and running forward pass
        '''

        # self.pruning_parameters[layer]["set_to_zero"][0]["parameter"]

        # stop adding data if needed
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        if self.method == 40:
            # for oracle let's try to do the best possible oracle by evaluating all neurons for each batch
            # self.oracle_dict["initial_loss"] += initial_loss
            self.oracle_dict["iterations"]   += 1

            # init_loss =
            outputs = model.forward(data)
            loss = criterion(outputs, target)
            init_loss = loss.item()

            print(f"==Pruning==oracle, intial loss: {init_loss}")

            # do first pass with precomputed values
            for layer_index, layer_parameters in enumerate(self.parameters):

                # print(f"===Pruning===oracle, layer: {layer_index}/{len(self.parameters)}")

                # start list of estimates for the layer if it is empty
                if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                    self.oracle_dict["loss_list"].append(list())

                if not self.prune_layers[layer_index]:
                    continue
                # copy original prune_layer variable that sets layers to be prunned
                self.prune_layers_oracle = [False, ]*len(self.parameters)
                self.prune_layers_oracle[layer_index] = True
                # store weights for future to recover
                self.stored_weights = deepcopy(self.parameters[layer_index].data.cpu().numpy())
                # import pdb; pdb.set_trace()

                for neurion_id, neuron in enumerate(layer_parameters):
                    # set neuron to zero
                    self.oracle_unit = neurion_id
                    self.set_weights_oracle_pruning()

                    # if self.stored_weights[neurion_id].sum() == 0.0:
                    #     new_loss = 0.0
                    # else:
                    if 1:
                        outputs = model.forward(data)
                        loss = criterion(outputs, target)
                        new_loss = loss.item()

                    # define loss, will be KL between 2 distributions

                    oracle_value = new_loss

                    #should be a difference from initial loss
                    # oracle_value = new_loss - init_loss

                    # print(f"==Pruning==oracle, unit {neurion_id}, loss: {new_loss}")

                    # pdb.set_trace()

                    oracle_value = abs(init_loss - new_loss)
                    # relative loss for testing:
                    # oracle_value = initial_loss - new_loss

                    if len(self.oracle_dict["loss_list"][layer_index]) == 0:
                        self.oracle_dict["loss_list"][layer_index] = [oracle_value, ]
                    elif len(self.oracle_dict["loss_list"][layer_index]) < neurion_id+1:
                        self.oracle_dict["loss_list"][layer_index].append(oracle_value)
                    else:
                        self.oracle_dict["loss_list"][layer_index][neurion_id] += oracle_value

                stats = np.asarray(self.oracle_dict["loss_list"][layer_index])
                print(f"===Pruning===oracle, layer: {layer_index}/{len(self.parameters)}, stats: min {stats.min()}, max {stats.max()}, mean {stats.mean()}")


                self.reset_oracle_pruning()

            import pickle
            with open(self.log_folder + "/oracle_temp_bn_rel.p","wb") as f:
                pickle.dump(self.oracle_dict, f)

    def compute_mask_overlap(self, old_list, new_list):
        #calculate how many previous neurons are in the current mask
        intersection = 0
        for layer_id in range(len(old_list)):
            intersection += ((1.0-old_list[layer_id])*(1.0-new_list[layer_id])).sum()
            # print("%02d"%layer_id, "\t", intersection, "\t", (1.0-old_list[layer_id]).sum(), "\t", (1.0-new_list[layer_id]).sum())

        total_number = sum([(1.0-a).sum() for a in old_list])

        matching_score = (intersection + 1.0)/(total_number + 1.0)

        return matching_score

    def compose_loss(self):

        power_rank = 2
        # power_rank = 1
        SHOW_ONCE=True


        loss = torch.zeros(1, requires_grad=True).cuda()
        # print("loss --->" , loss)
        # data_list = list()

        # data_list = torch.cat(self.parameters).pow(2)
        data_list = []

        if not self.push_down:
            return loss, data_list

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            zeroed_el = np.nonzero(1.0 - self.pruning_gates[layer])[0]

            # print("layer %02d: "%layer,"\t",len(zeroed_el),"\t/\t",len(self.pruning_gates[layer]))

            if len(zeroed_el) == 0:
                continue
            if 0:
                #go with corresponding parameters
                for param in self.pruning_parameters[layer]["set_to_zero"]:
                    set_to_zero = zeroed_el + param["shift"]
                    if check_allow_trim(param):
                        in_the_range = zeroed_el + param["shift"] < param["parameter"].data.shape[param["dim"]]
                        in_the_range = in_the_range * (zeroed_el + param["shift"] >= 0)
                        set_to_zero = set_to_zero[in_the_range]

                        # print("allow_trim", in_the_range, unit + param["shift"], param["parameter"].data.shape[param["dim"]], self.pruning_gates[layer].shape)

                    if param["dim"] == 0:
                        if power_rank == 2:
                            weight_mag_loss = param["parameter"][set_to_zero].pow(power_rank).sum()
                        else:
                            weight_mag_loss = param["parameter"][set_to_zero].abs().sum()
                    elif param["dim"] == 1:
                        if power_rank == 2:
                            weight_mag_loss = param["parameter"][:,set_to_zero].pow(power_rank).sum()
                        else:
                            weight_mag_loss = param["parameter"][:, set_to_zero].abs().sum()

                    loss += weight_mag_loss
            else:
                #go with only from where we compute criterion
                zeroed_el = np.nonzero(1.0 - self.pruning_gates[layer])[0]
                # print(len(self.pruning_parameters[layer]["compute_criteria_from"]))
                for w_ind, w in enumerate(self.pruning_parameters[layer]["compute_criteria_from"]):

                    value = w["parameter"][zeroed_el]
                    #mimic l2
                    # weight_update_decay = 0.5 * abs(value.data) * self.push_down_weight_decay

                    # value.data = value.data - abs(value.data)*self.push_down_weight_decay
                    pruning_mask = torch.from_numpy((1.0 - self.pruning_gates[layer])).cuda().type(w["parameter"].data.type())

                    w["parameter"].data = w["parameter"].data - pruning_mask*self.push_down_weight_decay*w["parameter"].data
                    # print(self.push_down_weight_decay*w["parameter"].data)
                    # if layer==1:
                    # if SHOW_ONCE and len(zeroed_el)>0:
                    #     print("weight decay, ",self.push_down_weight_decay)
                    #     print("weight      , ",w["parameter"])
                    #     print("gradients     ",value.grad)
                    #     # print((1.0 - value.grad.abs() / w["parameter"].grad.abs().argmax() ))
                    #     SHOW_ONCE = False
                    # weight_update_decay - for decay
                    # value.grad - for gradient
                    # w["parameter"].grad.max()
                    # w["parameter"].grad.abs().argmax()
                    # (value.grad.abs() / w["parameter"].grad.abs().argmax() ) # 1 - if it is the max, don't penalize #0 -means doesn't matter penalize
                    # multiplier, if 1 means prune more, if 0 - better don't touch:
                    # (1.0 - value.grad.abs() / w["parameter"].grad.abs().argmax() )

                    #check gradients, if gradient is very large for this neuron then don't penalize it?
                    #we don't want to overparamtrize it here
                    #PROBLEM STATEMENT, how to prune with not overpruning the main loss, check gradient from the loss and find relationship

                    loss += value.abs().sum()

                # data_list.append(weight_mag_loss.data.cpu().numpy())

        # print("total on weight decay: ", sum([(a==0.0).sum() for a in self.pruning_gates]))
        # print("loss2 --->", loss)
        return loss, data_list

    def compute_mask_overlap(self, old_list, new_list):
        #calculate how many previous neurons are in the current mask
        intersection = 0
        for layer_id in range(len(old_list)):
            intersection += ((1.0-old_list[layer_id])*(1.0-new_list[layer_id])).sum()
            # print("%02d"%layer_id, "\t", intersection, "\t", (1.0-old_list[layer_id]).sum(), "\t", (1.0-new_list[layer_id]).sum())

        total_number = sum([(1.0-a).sum() for a in old_list])

        matching_score = (intersection + 1.0)/(total_number + 1.0)

        return matching_score

    def compose_loss(self):

        power_rank = 2
        # power_rank = 1
        SHOW_ONCE=True


        loss = torch.zeros(1, requires_grad=True).cuda()
        # print("loss --->" , loss)
        # data_list = list()

        data_list = torch.cat(self.parameters).pow(2)

        if not self.push_down:
            return loss, data_list

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            zeroed_el = np.nonzero(1.0 - self.pruning_gates[layer])[0]

            # print("layer %02d: "%layer,"\t",len(zeroed_el),"\t/\t",len(self.pruning_gates[layer]))

            if len(zeroed_el) == 0:
                continue
            if 0:
                #go with corresponding parameters
                for param in self.pruning_parameters[layer]["set_to_zero"]:
                    set_to_zero = zeroed_el + param["shift"]
                    if check_allow_trim(param):
                        in_the_range = zeroed_el + param["shift"] < param["parameter"].data.shape[param["dim"]]
                        in_the_range = in_the_range * (zeroed_el + param["shift"] >= 0)
                        set_to_zero = set_to_zero[in_the_range]

                        # print("allow_trim", in_the_range, unit + param["shift"], param["parameter"].data.shape[param["dim"]], self.pruning_gates[layer].shape)

                    if param["dim"] == 0:
                        if power_rank == 2:
                            weight_mag_loss = param["parameter"][set_to_zero].pow(power_rank).sum()
                        else:
                            weight_mag_loss = param["parameter"][set_to_zero].abs().sum()
                    elif param["dim"] == 1:
                        if power_rank == 2:
                            weight_mag_loss = param["parameter"][:,set_to_zero].pow(power_rank).sum()
                        else:
                            weight_mag_loss = param["parameter"][:, set_to_zero].abs().sum()

                    loss += weight_mag_loss
            else:
                #go with only from where we compute criterion
                zeroed_el = np.nonzero(1.0 - self.pruning_gates[layer])[0]
                for w_ind, w in enumerate(self.pruning_parameters[layer]["compute_criteria_from"]):

                    value = w["parameter"][zeroed_el]
                    #mimic l2
                    # weight_update_decay = 0.5 * abs(value.data) * self.push_down_weight_decay

                    # value.data = value.data - abs(value.data)*self.push_down_weight_decay
                    pruning_mask = torch.from_numpy((1.0 - self.pruning_gates[layer])).cuda().type(w["parameter"].data.type())

                    w["parameter"].data = w["parameter"].data - pruning_mask*self.push_down_weight_decay*w["parameter"].data

                    if 0:
                        if SHOW_ONCE and len(zeroed_el)>0:
                            # print("weight decay, ",weight_update_decay)
                            # print("weight      , ",w["parameter"])
                            # print("gradients     ",value.grad)
                            # print((1.0 - value.grad.abs() / w["parameter"].grad.abs().argmax() ))
                            SHOW_ONCE = False
                    # weight_update_decay - for decay
                    # value.grad - for gradient
                    # w["parameter"].grad.max()
                    # w["parameter"].grad.abs().argmax()
                    # (value.grad.abs() / w["parameter"].grad.abs().argmax() ) # 1 - if it is the max, don't penalize #0 -means doesn't matter penalize
                    # multiplier, if 1 means prune more, if 0 - better don't touch:
                    # (1.0 - value.grad.abs() / w["parameter"].grad.abs().argmax() )

                    #check gradients, if gradient is very large for this neuron then don't penalize it?
                    #we don't want to overparamtrize it here
                    #PROBLEM STATEMENT, how to prune with not overpruning the main loss, check gradient from the loss and find relationship

                    loss += value.abs().sum()

                # data_list.append(weight_mag_loss.data.cpu().numpy())

        # print("total on weight decay: ", sum([(a==0.0).sum() for a in self.pruning_gates]))
        # print("loss2 --->", loss)
        return loss, data_list





class ExpMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, mom = 0.9):
        self.reset()
        self.mom = mom

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean_avg = self.sum / self.count
        self.exp_avg = self.mom*self.exp_avg + (1.0 - self.mom)*self.val
        if self.count == 1:
            self.exp_avg = self.val

if __name__ == '__main__':
    pass
        