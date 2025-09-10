### Will create trace for pruning
import torch
import pdb


def initilize_layer_pruning(layer, extra_name="", dim = 0):
    # initializes pruning of particular layer by specifying relevant fields.
    # Later, the pruner will iterate over modules and will prune it if sees .do_pruning = True
    # Parameters from which to compute criteria are stored in the field compute_criteria_from
    # Parameters that needs to be pruned are stored in set_to_zero
    # There are other ways to specify what to prune.
    layer.do_pruning = True
    compute_criteria_from = [{"parameter_name": extra_name + ".weight:" + layer.__repr__(), "dim": dim, "parameter": layer.weight, "fix": True}, ]

    set_to_zero = [{"parameter_name": extra_name + ".weight:" + layer.__repr__(), "dim": dim, "parameter": layer.weight}]

    if dim == 0:
        if hasattr(layer, "bias"):
            if not ((layer.bias is False) or (layer.bias is None)):
                compute_criteria_from.append(
                    {"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 0, "parameter": layer.bias})
                set_to_zero.append({"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 0, "parameter": layer.bias})
                
    if dim == 2:
        if hasattr(layer, "bias"):
            if not ((layer.bias is False) or (layer.bias is None)):
                compute_criteria_from.append(
                    {"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 1, "parameter": layer.bias})
                set_to_zero.append({"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 1, "parameter": layer.bias})

    layer.compute_criteria_from = compute_criteria_from
    layer.set_to_zero = set_to_zero


def add_compute_criteria_names(layer, name):
    if hasattr(layer, "compute_criteria_from_names"):
        layer.compute_criteria_from_names.append(name)
    else:
        layer.compute_criteria_from_names = [name, ]


def add_set_to_zero_names(layer, name):
    if hasattr(layer, "compute_criteria_from_names"):
        layer.set_to_zero_names.append(name)
    else:
        layer.set_to_zero_names = [name, ]


def connect_output_to_input(parent_parameter, child_parameter, dim=0, shift=0,bias=True, extra_name="", allow_trim=False):
    # connects pruning to zero out dependent child layers
    # parent_parameter - main layer from which to compute statistics
    # child_parameter  - parameter to be zeroed depending on the loss from parent
    # dim - dimension to be affected, 0 - output channel, 1 - input channel
    # bias - set to zero bias as well
    parent_parameter.set_to_zero.append({"parameter_name": extra_name  + ".weight:"+ child_parameter.__repr__(),
                                         "dim": dim, "parameter": child_parameter.weight, "shift": shift, "allow_trim": allow_trim})
    if hasattr(child_parameter, 'bias') and dim==0:
        if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
            parent_parameter.set_to_zero.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                 "dim": dim, "parameter": child_parameter.bias})
    if hasattr(child_parameter, 'bias') and dim==2 and bias:
        if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
            parent_parameter.set_to_zero.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                 "dim": 1, "parameter": child_parameter.bias})


def link_criteria_layers(parent_parameter, child_parameter, dim=0, extra_name=""):
    parent_parameter.compute_criteria_from.append({"parameter_name": extra_name  + ".weight:"+ child_parameter.__repr__(),
                                                   "dim": dim, "parameter": child_parameter.weight, "layer_link": child_parameter})
    if dim==0:
        if hasattr(child_parameter, 'bias'):
            if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
                parent_parameter.compute_criteria_from.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                           "dim": dim, "parameter": child_parameter.bias})
    if dim==2:
        if hasattr(child_parameter, 'bias'):
            if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
                parent_parameter.compute_criteria_from.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                           "dim": 1, "parameter": child_parameter.bias})


def create_pruning_structure_vit(student, prune_token=False, prune_emb=False, prune_MLP=False, prune_head=False, prune_qk=False, prune_v=False, only_skip = False):
    first_layer = True
    # from IPython import embed; embed()
    for name, par in student.named_modules():
        #print(name)
        layer = par
        if not only_skip:
            # all experiments so far are with this state
            if prune_token:
                if ".attn.qkv" in name and "qkv." not in name:
                    layer.do_pruning = True
                    layer.fix = True
                    compute_criteria_from = [{"parameter_name": name + ".mask:" + layer.__repr__(), "dim": 0, "parameter": layer.token_mask}, ]
                    set_to_zero = [{"parameter_name": name + ".mask:" + layer.__repr__(), "dim": 0, "parameter": layer.token_mask}]
                    layer.compute_criteria_from = compute_criteria_from
                    layer.set_to_zero = set_to_zero
                    
            if prune_emb:
                if "patch_embed.proj" in name:
                    initilize_layer_pruning(layer, extra_name="{}_EMB".format(name))
                    last_proj_layer = layer
                    last_proj_layer.set_to_zero.append({"parameter_name": "{}_EMB".format(name)  + ".weight:cls_token",
                                         "dim": 2, "parameter": student.module.cls_token, "shift": 0, "allow_trim": False})
                    last_proj_layer.set_to_zero.append({"parameter_name": "{}_EMB".format(name)  + ".weight:pos_embed",
                                         "dim": 2, "parameter": student.module.pos_embed, "shift": 0, "allow_trim": False})
                    last_proj_layer.set_to_zero.append({"parameter_name": "{}_EMB".format(name)  + ".weight:dist_token",
                                         "dim": 2, "parameter": student.module.dist_token, "shift": 0, "allow_trim": False})
                if "norm" in name:
                    connect_output_to_input(last_proj_layer, layer, dim=0, extra_name="{}_EMB".format(name))
                if ".attn.proj" in name and "drop" not in name:
                    link_criteria_layers(last_proj_layer, layer, dim=0, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=0, extra_name="{}_EMB".format(name))
                if ".mlp.fc1" in name:
                    link_criteria_layers(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                if ".mlp.fc2" in name:
                    link_criteria_layers(last_proj_layer, layer, dim=0, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=0, extra_name="{}_EMB".format(name))
                if ".attn.qkv.Q" in name:
                    link_criteria_layers(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                if ".attn.qkv.K" in name:
                    link_criteria_layers(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                if ".attn.qkv.V" in name:
                    link_criteria_layers(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                if "head" in name and "blocks" not in name:
                    #link_criteria_layers(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                    connect_output_to_input(last_proj_layer, layer, dim=1, extra_name="{}_EMB".format(name))
                
                
            if prune_MLP:
                if ".mlp.fc1" in name:
                    #add forward pruning layer
                    initilize_layer_pruning(layer, extra_name="{}_".format(name))
                    last_mlp_layer = layer
                if ".mlp.fc2" in name:
                    #this layer is pruned in correspondence to another layer we pruned before
                    connect_output_to_input(last_mlp_layer, layer, dim=1,
                                            extra_name="{}_".format(name))

            if prune_head:
                if ".attn.qkv.head_mask" in name:
                    #add forward pruning layer
                    initilize_layer_pruning(layer, extra_name="{}_".format(name),dim=0)
                    last_head_layer = layer
                if ".attn.qkv.Q" in name:
                    #add forward pruning layer
                    link_criteria_layers(last_head_layer, layer, dim=2,
                                            extra_name="{}_".format(name))
                    connect_output_to_input(last_head_layer, layer, dim=2,
                                            extra_name="{}_".format(name))
                if ".attn.qkv.K" in name:
                    #add forward pruning layer
                    link_criteria_layers(last_head_layer, layer, dim=2,
                                            extra_name="{}_".format(name))
                    connect_output_to_input(last_head_layer, layer, dim=2,
                                            extra_name="{}_".format(name))
                if ".attn.qkv.V" in name:
                    #add forward pruning layer
                    link_criteria_layers(last_head_layer, layer, dim=2,
                                            extra_name="{}_".format(name))
                    connect_output_to_input(last_head_layer, layer, dim=2,
                                            extra_name="{}_".format(name))
                if ".attn.proj" in name and "drop" not in name:
                    #add forward pruning layer
                    connect_output_to_input(last_head_layer, layer, dim=2,bias=False,
                                            extra_name="{}_".format(name))
            
            if prune_qk:
                if ".attn.qkv.Q" in name:
                    #add forward pruning layer
                    initilize_layer_pruning(layer, extra_name="{}_".format(name))
                    last_q_layer = layer
                if ".attn.qkv.K" in name:
                    #add forward pruning layer
                    connect_output_to_input(last_q_layer, layer, dim=0,
                                            extra_name="{}_".format(name))

            if prune_v:
                if ".attn.qkv.V" in name:
                    #add forward pruning layer
                    initilize_layer_pruning(layer, extra_name="{}_".format(name))
                    last_v_layer = layer
                if ".attn.proj" in name and "drop" not in name:
                    #add forward pruning layer
                    #initilize_layer_pruning(layer, extra_name="{}_".format(name), dim=1)
                    connect_output_to_input(last_v_layer, layer, dim=1,
                                        extra_name="{}_".format(name))

        else:
            # pruning skip connection only
            if "layernorm" in name:
                if first_layer:
                    initilize_layer_pruning(layer, extra_name="{}_".format(name), dim=0)
                    hidden_layer=layer
                    first_layer=False
                else:
                    link_criteria_layers(hidden_layer, layer, dim=0,
                                         extra_name="{}_".format(name))
                    connect_output_to_input(hidden_layer, layer, dim=0,
                                            extra_name="{}_".format(name))

def enable_pruning(pruning_engine, prune_token=False, prune_emb=False, prune_MLP=False, prune_head=False, prune_qk=False, prune_v=False, only_skip = False):
    first_layer = True
    for layer, if_prune in enumerate(pruning_engine.prune_layers):
        if not if_prune:
            continue
        pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix']=True
        name = pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]["parameter_name"]
        
        if pruning_engine.use_momentum and len(pruning_engine.prune_network_accomulate["averaged"][layer]):
            pruning_engine.prune_network_accomulate["averaged"][layer] *= 0.0
        
        if not only_skip:
            # all experiments so far are with this state
            if prune_token:
                if ".attn.qkv" in name and "qkv." not in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_emb:
                if "patch_embed.proj" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_MLP:
                if ".mlp.fc1" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_head:
                if ".attn.qkv.head_mask" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_qk:
                if ".attn.qkv.Q" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_v:
                if ".attn.qkv.V" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False

        else:
            pass
