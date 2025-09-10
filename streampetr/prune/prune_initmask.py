from mmcv.runner import Hook
import torch.nn as nn
import torch
import json
import pickle as pkl
from prune.pruning import update_neuron_metric, update_neuron_metric_hahp
from prune.hahp_func import *
from prune.latency_targeted_pruning_hahp_latv2 import apply_latency_target_pruning, get_total_latency, set_latency_prune_target

def extract_layers(model, layers={}, pre_name='', get_conv=True, get_bn=False, get_gate=False):
    """
        Get all the model layers and the names of the layers
    Returns:
        layers: dict, the key is layer name and the value is the corresponding model layer
    """
    for name, layer in model.named_children():
        new_name = '.'.join([pre_name,name]) if pre_name != '' else name
        if len(list(layer.named_children())) > 0:
            extract_layers(layer, layers, new_name, get_conv, get_bn, get_gate)
        else:
            get_layer = False
            if get_conv and is_conv(layer):
                get_layer = True
            elif get_bn and is_bn(layer):
                get_layer = True
            elif get_gate and 'gate' in new_name:
                get_layer = True
            if get_layer:
                layers[new_name] = layer
    return layers


def is_conv(layer):
    conv_type = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    isConv = False
    for ctp in conv_type:
        if isinstance(layer, ctp):
            isConv = True
            break
    return isConv


def is_bn(layer):
    bn_type = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    isbn = False
    for ctp in bn_type:
        if isinstance(layer, ctp):
            isbn = True
            break
    return isbn

class PruneHookInitMask(Hook):
    def __init__(self, model, result_dir, mask_path):
        # initialize your class variables here
        super().__init__()
        from prune.prune_config import PruneConfigReader
        config_reader = PruneConfigReader()
        config_reader.set_prune_setting("./prune/configs/resnet50.json")
        layer_structure = config_reader.get_layer_structure()  # conv_bn, conv_gate, groups
        conv_bn, conv_gate, groups = layer_structure
        from prune.prune_config_with_structure import PruneConfigReader
        config_reader = PruneConfigReader()
        config_reader.set_prune_setting("./prune/configs/resnet50_structure.json")
        _, _, _, pre_group, aft_group_list = config_reader.get_layer_structure()
        
        with open("./prune/cudnn_v7.4_conv_LUT_repeat_100_step_2_scope_2048_batch256_forward_resnet50_ngc.pkl", 'rb') as f:
            self.lookup_table = pkl.load(f)

        self._model = model
        self.result_dir = result_dir
        self.layers = extract_layers(model, get_conv=True, get_bn=True, get_gate=False)
        self.layers = {f"module.{k}":v for k,v in self.layers.items()}
        with open('./prune/net_structure/{}_fmap.json'.format("resnet50"), 'r') as f:
            self.fmap_table = json.load(f)
        from prune.latency_targeted_pruning_hahp_latv2 import load_group_size
        load_group_size("resnet50")
        print("Finished loading channel group size")
        
        # print(self.layers.keys())
        # exit()
        self.aft_group_list = aft_group_list
        self.pre_group = pre_group
        self.groups = groups
        self.layer_bn = conv_bn
        # Start to compute importance score
        self.imp_start = 50
        self.method = 26
        self.lut_bs = 256
        self.prune_steps = 1
        # self.start_prune = 930
        self.start_prune = 2000
        # self.start_prune = 60
        self.prune_ratio = 0.8
        self.prune_mode = "exp2"
        self.neuron_metric = {}
        # Initialize the mask
        # self.layer_masks = {}
        # for group in groups:
        #     for layer_name in group:
        #         self.layer_masks[layer_name] = torch.ones(self.layers[layer_name].weight.size(0)).to(self.layers[layer_name].weight.device)
        print("Loading the mask")
        with open(mask_path, 'rb') as f:
            self.layer_masks = pkl.load(f)
        for group in groups:
            for layer_name in group:
                self.layer_masks[layer_name] = self.layer_masks[layer_name].to(self.layers[layer_name].weight.device)
                # self.layer_masks[layer_name] = torch.zeros(self.layers[layer_name].weight.size(0)).to(self.layers[layer_name].weight.device)


    def mask_weights(self):
        for layer_name, mask in self.layer_masks.items():
            weight = self.layers[layer_name].weight
            # assert weight.grad.data[mask==0.].sum().item() == 0.
            weight.grad.data.mul_(mask.view(-1, 1, 1, 1))
            weight.data.mul_(mask.view(-1, 1, 1, 1))
            param_state = self.optimizer.state[weight]
            if 'momentum_buffer' in param_state:
                param_state['momentum_buffer'].data.mul_(mask.view(-1, 1, 1, 1))
            bias = self.layers[layer_name].bias
            if bias is not None:
                bias.grad.data.mul_(mask)
                bias.data.mul_(mask)
                param_state = self.optimizer.state[bias]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].data.mul_(mask)

            if self.layer_bn is not None:
                self.layers[self.layer_bn[layer_name]].weight.data.mul_(mask)
                # self.layers[self.layer_bn[layer_name]].weight.grad.data.mul_(mask)
                self.layers[self.layer_bn[layer_name]].bias.data.mul_(mask)
                # self.layers[self.layer_bn[layer_name]].bias.grad.data.mul_(mask)
                # param_state = self.optimizer.state[self.layers[self.layer_bn[layer_name]].weight]
                # if 'momentum_buffer' in param_state:
                #     param_state['momentum_buffer'].data.mul_(mask)
                # param_state = self.optimizer.state[self.layers[self.layer_bn[layer_name]].bias]
                # if 'momentum_buffer' in param_state:
                #     param_state['momentum_buffer'].data.mul_(mask)

    def turn_off_BN_grad(self):
        for layer_name, layer in self.layers.items():
            if isinstance(layer, nn.BatchNorm2d):
                # print("turn off", layer_name)
                # print(layer.requires_grad)
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                # layer.eval()

    def prepare_pruning(self):
        initial_latency = get_total_latency(self.layers, self.groups, self.pre_group, self.layer_masks, self.fmap_table, self.lookup_table, self.lut_bs)
        prune_target = set_latency_prune_target(initial_latency, 0, self.prune_steps, None, self.prune_ratio, mode=self.prune_mode)
        print(f"Initial Latency: {initial_latency}")
        print(prune_target)
        self.prune_target = prune_target

    def after_train_iter(self, runner):
        # your code here
        # model = runner.model
        # backbone = model.module.img_backbone
        # for name, layer in self.layers.items():
        #     print(name)
        # for name, module in runner.model.module.img_backbone.named_modules():
        #     if "bn" in name:
        #         grad = module.weight.grad
        #         print("after", grad)
        cur_iter = runner.iter
        self.optimizer = runner.optimizer
        if cur_iter == 0:
            self.prepare_pruning()
        # Update importance
        # if cur_iter > self.imp_start:
        #     update_neuron_metric_hahp(self.layers, self.groups, self.layer_bn, neuron_metric=self.neuron_metric)
        # if cur_iter == self.start_prune:
        #     # Perform pruning
        #     print("Start pruning")
        #     for layer_name in self.neuron_metric.keys():
        #         self.neuron_metric[layer_name] = torch.div(self.neuron_metric[layer_name], cur_iter - self.imp_start)
            
        #     blocks = build_block_from_layers(self.layers)
        #     output_dir = "./"
        #     layer_gate = None
        #     target_latency = self.prune_target.pop(0)
        #     apply_latency_target_pruning(output_dir, self.layers, self.layer_bn, blocks, layer_gate, self.neuron_metric, target_latency, self.layer_masks,\
        #             self.groups, self.pre_group, self.aft_group_list, self.fmap_table, self.lookup_table, self.method, \
        #             wrap_solver=None, mu=0., step_size=-1, lut_bs=256, pulp=False, blocks_num=None, no_blockprune=False)

        #     with open(f"{self.result_dir}/layer_masks.pkl", 'wb') as f:
        #         pkl.dump(self.layer_masks, f)
        #     # exit()
        self.mask_weights()
