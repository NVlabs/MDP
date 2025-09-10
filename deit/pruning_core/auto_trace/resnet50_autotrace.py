import torch
from torch.autograd import Variable

from pruning_core.pruning_utils import connect_output_to_input, initilize_layer_pruning, link_criteria_layers

if __name__=="__main__":
    import torch
    import torchvision.models as models
    import pdb

    # creating a resnet 50 model:
    resnet50 = models.resnet50(pretrained=True)

    image_batch = torch.empty((2,3,224,224)).normal_()

    import sys
    sys.path.append(".")
    from pruning_core.auto_trace.torch_pruning.dependency import DependencyGraph

    DG = DependencyGraph()
    DG.build_dependency(resnet50, example_inputs=torch.randn(1, 3, 112, 112))

    ### display dependencies
    if 0:
        for module, node in DG.module_to_node.items():
            if isinstance(module, torch.nn.BatchNorm2d):
                if len(node.dependencies) > 0:
                    print(node.details())


    for module, node in DG.module_to_node.items():
        if isinstance(module, torch.nn.BatchNorm2d):
            if len(node.dependencies) > 0:
                # if ("bn1" in node._node_name) or ("bn2" in node._node_name):
                #     continue
                # prefiltering

                print(f"Initialize pruning for {node._node_name}")
                initilize_layer_pruning(module, parameter_name=node._node_name)

                for dep in node.dependencies:
                    if dep.type == "in":
                        print("Connecting input ", node._node_name ," ", dep.node._node_name, " dim=", 0)
                        print(node.module.weight.shape, dep.node.module.weight.shape)
                        connect_output_to_input(module, dep.node.module, dim=0, parameter_name=dep.node._node_name)
                    if dep.type == "out":
                        print("Connecting input ", node._node_name , " ", dep.node._node_name, " dim=", 1)
                        print(node.module.weight.shape, dep.node.module.weight.shape)
                        connect_output_to_input(module, dep.node.module, dim=1, parameter_name=dep.node._node_name)
                    if dep.type == "bro":
                        print("Connecting pruning layer ", node._node_name , " ", dep.node._node_name, " dim=", 0)
                        print(node.module.weight.shape, dep.node.module.weight.shape)
                        connect_output_to_input(module, dep.node.module, dim=0, parameter_name=dep.node._node_name)
                        link_criteria_layers(module, dep.node.module, dim=0)


