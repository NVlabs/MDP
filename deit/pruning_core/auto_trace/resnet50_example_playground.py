import torch
import torchvision.models as models
import pdb

# creating a resnet 50 model:
resnet50 = models.resnet50(pretrained=True)

image_batch = torch.empty((16,3,224,224)).normal_()

traced_resnet50 = torch.jit.trace(resnet50, image_batch)
script_resnet50 = torch.jit.script(resnet50, image_batch)
trace, grad = torch.jit._get_trace_graph(resnet50, image_batch) ##more information

# print(traced_resnet50.graph)

script_resnet50.save("model_script.ts")

with open("graph.txt",'w') as f:
    f.write("{}".format(traced_resnet50.graph))
with open("code.txt",'w') as f:
    f.write("{}".format(traced_resnet50.code))
with open("full_graph.txt",'w') as f:
    f.write("{}".format(trace))


if 1:
    #use torchviz
    #https://github.com/szagoruyko/pytorchviz
    #sudo apt-get install graphviz
    import sys
    sys.path.append(".")
    from pruning_core.auto_trace.torchviz import make_dot, make_dot_from_trace

    model_arch = make_dot(resnet50(image_batch), params=dict(resnet50.named_parameters()))
    from graphviz import Source

    filepath = "graph.png"
    Source(model_arch).render(filepath, format = "png")

    # for att,v in model_arch.graph_attr():
    #     print(att, v)

    # traced_resnet50 = torch.jit.trace(resnet50, image_batch)
    #
    # dot = make_dot_from_trace(traced_resnet50)
    # filepath = "graph2.png"
    # Source(dot).render(filepath)
    
    # print(md)
    pdb.set_trace()

if 1:
    #use tensorboard to display the model graph
    #pip install tensorboard
    #https://pytorch.org/docs/stable/_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_graph
    from torch.utils.tensorboard import SummaryWriter
    # from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir="./temp")
    writer.add_graph(resnet50, image_batch, verbose=True)
    writer.close()

# for node in trace.nodes():
#     print("*" * 40)
#     print(node)
#     pdb.set_trace()





if 0:
    ### with TRT
    import trtorch

    script_resnet50.eval() # torch module needs to be in eval (not training) mode

    compile_settings = {
        "input_shapes": [
            {
                "min": [16, 3, 224, 224],
                "opt": [16, 3, 224, 224],
                "max": [16, 3, 224, 224]
            },
        ],
        "op_precision": torch.half # Run with fp16
    }

    trt_ts_module = trtorch.compile(script_resnet50, compile_settings)

    input_data = image_batch.to('cuda').half()
    result = trt_ts_module(input_data)
    # torch.jit.save(trt_ts_module, "trt_ts_module.ts")

if 0:
    import torch.onnx
    dummy_input = Variable(torch.randn(4, 3, 32, 32))
    torch.onnx.export(net, dummy_input, "model.onnx")



#https://github.com/VainF/Torch-Pruning/blob/master/torch_pruning/dependency.py
## https://ad1024.space/articles/22
## NVIDIA TRT on JIT scripts and trace: https://nvidia.github.io/TRTorch/tutorials/getting_started.html
## example fo trace creation with nvPruner
