from torch import nn
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, num_class=1000):
        super(Net, self).__init__()
        self.num_class = num_class

        def conv_bn(inp, oup, stride):
            layers = [('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
                      ('bn', nn.BatchNorm2d(oup)),
                      ('relu', nn.ReLU(inplace=True))]
            return nn.Sequential(OrderedDict(layers))

        def conv_dw(inp, oup, stride):
            layers = [('conv1', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)),
                      ('bn1', nn.BatchNorm2d(inp)),
                      ('relu1', nn.ReLU(inplace=True)),
                      ('conv2', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                      ('bn2', nn.BatchNorm2d(oup)),
                      ('relu2', nn.ReLU(inplace=True))]
            return nn.Sequential(OrderedDict(layers))
        
        layers = [('conv_bn', conv_bn(3, 32, 2)),
                  ('conv_dw1', conv_dw(32, 64, 1)),
                  ('conv_dw2', conv_dw(64, 128, 2)),
                  ('conv_dw3', conv_dw(128, 128, 1)),
                  ('conv_dw4', conv_dw(128, 256, 2)),
                  ('conv_dw5', conv_dw(256, 256, 1)),
                  ('conv_dw6', conv_dw(256, 512, 2)),
                  ('conv_dw7', conv_dw(512, 512, 1)),
                  ('conv_dw8', conv_dw(512, 512, 1)),
                  ('conv_dw9', conv_dw(512, 512, 1)),
                  ('conv_dw10', conv_dw(512, 512, 1)),
                  ('conv_dw11', conv_dw(512, 512, 1)),
                  ('conv_dw12', conv_dw(512, 1024, 2)),
                  ('conv_dw13', conv_dw(1024, 1024, 1)),
                  ('avg_pool', nn.AvgPool2d(7))]
        self.features = nn.Sequential(OrderedDict(layers))
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenet(num_class=1000):
    model = Net(num_class)
    return model

