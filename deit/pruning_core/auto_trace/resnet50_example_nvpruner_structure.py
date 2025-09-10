import torch
import torchvision.models as models
import pdb
from pruning_core.pruning_utils import connect_output_to_input, initilize_layer_pruning
# work in progress

# creating a resnet 50 model:
resnet50 = models.resnet50(pretrained=True)

image_batch = torch.empty((16,3,224,224)).normal_()


pruning_parameters_list = list()
model = resnet50

# named_parameters = [(a, b) for a, b in list(model.named_parameters())]
# named_modules = [(a, b) for a, b in list(model.named_modules())]

# find_module = lambda name: \
#         [module for (module_name, module) in named_modules if module_name == name][0]

named_modules = dict(resnet50.named_modules())

for module_indx, (m_name, m_el) in enumerate(model.named_modules()):
    #asusme we prune only bn layers
    if isinstance(m_el, torch.nn.BatchNorm2d):
        # print(m_name)

        # for simplicity we prune only bottleneck layers, later we will want to prune all of them.
        # ideally we should not do it
        if not (("bn1" in m_name) or ("bn2" in m_name)):
            #skip those other layers
            continue

        initilize_layer_pruning(m_el, dim=0, parameter_name=m_name)

        set_to_zero = m_el.set_to_zero

        if "bn1" in m_name:
            # prune previous conv layer accordingly
            # conv layer weights are usually in the form of [CHANNELS_OUTPUT, CHANNELS_INPUT, WIDTH, HEIGHT], therefore the output dimension is 0, input is 1
            child_layer_name = m_name.replace("bn1", "conv1")
            child_layer = named_modules[child_layer_name]
            connect_output_to_input(parent_layer=m_el, child_parameter=child_layer, dim=0,
                                    parameter_name=child_layer_name)
            # prune next conv layer accordingly, dim=1
            child_layer_name = m_name.replace("bn1", "conv2")
            child_layer = named_modules[child_layer_name]
            connect_output_to_input(parent_layer=m_el, child_parameter=child_layer, dim=1,
                                    parameter_name=child_layer_name)
        if "bn2" in m_name:
            child_layer_name = m_name.replace("bn2", "conv2")
            child_layer = named_modules[child_layer_name]
            connect_output_to_input(parent_layer=m_el, child_parameter=child_layer, dim=0,
                                    parameter_name=child_layer_name)
            child_layer_name = m_name.replace("bn2", "conv3")
            child_layer = named_modules[child_layer_name]
            connect_output_to_input(parent_layer=m_el, child_parameter=child_layer, dim=1,
                                    parameter_name=child_layer_name)


        # We yet to add a pruning of skip connections and downsampling layers

