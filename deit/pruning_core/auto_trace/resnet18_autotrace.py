#launch from nvPruner folder:  python pruning_core/auto_trace/resnet18_autotrace.py
import sys
sys.path.append('.')
import torch
from torch.autograd import Variable
from pruning_core.pruning_utils import connect_output_to_input, initilize_layer_pruning, link_criteria_layers


if __name__=="__main__":
    import torch
    import torchvision.models as models
    import pdb

    # creating a resnet 50 model:
    resnet18 = models.resnet18(pretrained=True)

    image_batch = torch.empty((2,3,224,224)).normal_()

    import sys
    sys.path.append(".")
    from pruning_core.auto_trace.torch_pruning.dependency import DependencyGraph

    DG = DependencyGraph()
    DG.build_dependency(resnet18, example_inputs=torch.randn(1, 3, 112, 112))

    if 0:
        ### display dependencies
        for module, node in DG.module_to_node.items():
            if isinstance(module, torch.nn.BatchNorm2d):
                if len(node.dependencies) > 0:
                    print(node.details())

    # named_modules = dict(student.named_modules())
    # DG = DependencyGraph()
    # DG.build_dependency(resnet18, example_inputs=torch.randn(1, 3, 256, 192).cpu())

    for module, node in DG.module_to_node.items():
        if isinstance(module, torch.nn.BatchNorm2d):
            if len(node.dependencies) > 0:

                print(f"Initialize pruning for {node._node_name}")
                initilize_layer_pruning(module, parameter_name=node._node_name)

                for dep in node.dependencies:
                    if dep.type == "in" or dep.type == "bro": dim =0
                    if dep.type == "out": dim =1

                    print("Connecting input ", node._node_name, " ", dep.node._node_name, " dim=", dim)
                    print(node.module.weight.shape, dep.node.module.weight.shape)

                    connect_output_to_input(module, dep.node.module, dim=dim, parameter_name=dep.node._node_name)
                    if dep.type == "bro":
                        print(f"Linking layers {node._node_name} and {dep.node._node_name}. They are the same size and pruned together.")
                        link_criteria_layers(module, dep.node.module, dim=0)

