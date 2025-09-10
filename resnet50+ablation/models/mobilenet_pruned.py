from torch import nn
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, layer_num_dict, num_class=1000):
        super(Net, self).__init__()
        self.num_class = num_class
        self.layer_num_dict = layer_num_dict

        def conv_bn(inp, oup, stride):
            layers = [('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
                      ('bn', nn.BatchNorm2d(oup)),
                      ('relu', nn.ReLU(inplace=True))]
            return nn.Sequential(OrderedDict(layers))

        def conv_dw(inp, oup1, oup2, stride):
            layers = [('conv1', nn.Conv2d(inp, oup1, 3, stride, 1, groups=oup1, bias=False)),
                      ('bn1', nn.BatchNorm2d(oup1)),
                      ('relu1', nn.ReLU(inplace=True)),
                      ('conv2', nn.Conv2d(oup1, oup2, 1, 1, 0, bias=False)),
                      ('bn2', nn.BatchNorm2d(oup2)),
                      ('relu2', nn.ReLU(inplace=True))]
            return nn.Sequential(OrderedDict(layers))

        layers = [('conv_bn', conv_bn(3, layer_num_dict["module.features.conv_bn.conv"], 2)),
                  ('conv_dw1', conv_dw(layer_num_dict["module.features.conv_bn.conv"], layer_num_dict["module.features.conv_dw1.conv1"], layer_num_dict["module.features.conv_dw1.conv2"], 1)),
                  ('conv_dw2', conv_dw(layer_num_dict["module.features.conv_dw1.conv2"], layer_num_dict["module.features.conv_dw2.conv1"], layer_num_dict["module.features.conv_dw2.conv2"], 2)),
                  ('conv_dw3', conv_dw(layer_num_dict["module.features.conv_dw2.conv2"], layer_num_dict["module.features.conv_dw3.conv1"], layer_num_dict["module.features.conv_dw3.conv2"], 1)),
                  ('conv_dw4', conv_dw(layer_num_dict["module.features.conv_dw3.conv2"], layer_num_dict["module.features.conv_dw4.conv1"], layer_num_dict["module.features.conv_dw4.conv2"], 2)),
                  ('conv_dw5', conv_dw(layer_num_dict["module.features.conv_dw4.conv2"], layer_num_dict["module.features.conv_dw5.conv1"], layer_num_dict["module.features.conv_dw5.conv2"], 1)),
                  ('conv_dw6', conv_dw(layer_num_dict["module.features.conv_dw5.conv2"], layer_num_dict["module.features.conv_dw6.conv1"], layer_num_dict["module.features.conv_dw6.conv2"], 2)),
                  ('conv_dw7', conv_dw(layer_num_dict["module.features.conv_dw6.conv2"], layer_num_dict["module.features.conv_dw7.conv1"], layer_num_dict["module.features.conv_dw7.conv2"], 1)),
                  ('conv_dw8', conv_dw(layer_num_dict["module.features.conv_dw7.conv2"], layer_num_dict["module.features.conv_dw8.conv1"], layer_num_dict["module.features.conv_dw8.conv2"], 1)),
                  ('conv_dw9', conv_dw(layer_num_dict["module.features.conv_dw8.conv2"], layer_num_dict["module.features.conv_dw9.conv1"], layer_num_dict["module.features.conv_dw9.conv2"], 1)),
                  ('conv_dw10', conv_dw(layer_num_dict["module.features.conv_dw9.conv2"], layer_num_dict["module.features.conv_dw10.conv1"], layer_num_dict["module.features.conv_dw10.conv2"], 1)),
                  ('conv_dw11', conv_dw(layer_num_dict["module.features.conv_dw10.conv2"], layer_num_dict["module.features.conv_dw11.conv1"], layer_num_dict["module.features.conv_dw11.conv2"], 1)),
                  ('conv_dw12', conv_dw(layer_num_dict["module.features.conv_dw11.conv2"], layer_num_dict["module.features.conv_dw12.conv1"], layer_num_dict["module.features.conv_dw12.conv2"], 2)),
                  ('conv_dw13', conv_dw(layer_num_dict["module.features.conv_dw12.conv2"], layer_num_dict["module.features.conv_dw13.conv1"], layer_num_dict["module.features.conv_dw13.conv2"], 1)),
                  ('avg_pool', nn.AvgPool2d(7))]
        self.features = nn.Sequential(OrderedDict(layers))
        self.fc = nn.Linear(layer_num_dict["module.features.conv_dw13.conv2"], num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.layer_num_dict["module.features.conv_dw13.conv2"])
        x = self.fc(x)
        return x


def mobilenet(layer_num_dict, num_class=1000):
    model = Net(layer_num_dict, num_class)
    return model

