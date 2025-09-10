import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import math
from collections import OrderedDict


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    """
    VGG model
    """
    def __init__(self, features, dataset_name='CIFAR10'):
        super(VGG, self).__init__()
        self.features = features
        layers = []
        if dataset_name == 'CIFAR10':
            layers = [('linear1', nn.Linear(out_c, 512)),
                      ('relu1', nn.ReLU(True)),
                      ('dropout1', nn.Dropout()),
                      ('linear2', nn.Linear(512, 512)),
                      ('relu2', nn.ReLU(True)),
                      ('dropout2', nn.Dropout())]
        elif dataset_name == 'ImageNet':
            layers = [('linear1', nn.Linear(out_c * 7 * 7, 4096)),
                      ('relu1', nn.ReLU(True)),
                      ('dropout1', nn.Dropout()),
                      ('linear2', nn.Linear(4096, 4096)),
                      ('relu2', nn.ReLU(True)),
                      ('dropout2', nn.Dropout())]
        else:
            NotImplementedError('VGG for dataset {} is not implemented.'.format(dataset_name))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(OrderedDict(layers))
        if dataset_name == 'CIFAR10':
            self.fc = nn.Linear(512, 10)
        elif dataset_name == 'ImageNet':
            self.fc = nn.Linear(4096, 1000)
        # Initialize weights
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #        # m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1, 1, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def make_layers(cfg, batch_norm=False, bias=False):
    layers = []
    layer_counter = 1
    conv_counter = 1
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [('maxpool{}'.format(layer_counter), nn.MaxPool2d(kernel_size=2, stride=2))]
            layer_counter += 1
            conv_counter = 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                layers += [('conv{}-{}'.format(layer_counter, conv_counter), conv2d),
                           ('bn{}-{}'.format(layer_counter, conv_counter), nn.BatchNorm2d(v)),
                           ('relu{}-{}'.format(layer_counter, conv_counter), nn.ReLU(inplace=True))]
            else:
                layers += [('conv{}-{}'.format(layer_counter, conv_counter), conv2d),
                           ('relu{}-{}'.format(layer_counter, conv_counter), nn.ReLU(inplace=True))]
            conv_counter += 1
            in_channels = v
    return nn.Sequential(OrderedDict(layers))


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M']
}
out_c = 512


def set_cfg(layer_num_dict, cfg_key):
    i = 0
    for layer_name, neuron_num in layer_num_dict.items():
        if cfg[cfg_key][i] == 'M':
            i += 1
        cfg[cfg_key][i] = neuron_num
        i += 1


def vgg11(bias=False, dataset_name='CIFAR10'):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], bias=bias), dataset_name)


def vgg11_bn(bias=False, dataset_name='CIFAR10'):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True, bias=bias), dataset_name)


def vgg13(bias=False, dataset_name='CIFAR10'):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], bias=bias), dataset_name)


def vgg13_bn(bias=False, dataset_name='CIFAR10'):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True, bias=bias), dataset_name)


def vgg16(layer_num_dict, bias=False, dataset_name='CIFAR10'):
    """VGG 16-layer model (configuration "D")"""
    set_cfg(layer_num_dict, 'D')
    global out_c
    out_c = cfg['D'][-2]
    return VGG(make_layers(cfg['D'], bias=bias), dataset_name)


def vgg16_bn(bias=False, dataset_name='CIFAR10'):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True, bias=bias), dataset_name)


def vgg19(bias=False, dataset_name='CIFAR10'):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], bias=bias), dataset_name)


def vgg19_bn(bias=False, dataset_name='CIFAR10'):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, bias=bias), dataset_name)
