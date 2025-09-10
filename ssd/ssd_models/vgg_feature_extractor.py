import torch.nn as nn
from layers.gate_layer import GateLayer
from collections import OrderedDict
import torchvision


__all__ = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'make_gated_layer'
]


def make_layers(cfg, batch_norm=False, bias=False, no_gating=False, init_method='xavier'):
    """Return VGG feature extractor before maxpool4"""
    layers = []
    layer_counter = 1
    conv_counter = 1
    in_channels = 3
    maxpool_counter = 1

    # Layers until conv4_3
    for v in cfg:
        if v == 'M':
            if maxpool_counter == 4:
                break
            if maxpool_counter == 3:
                layers += [('maxpool{}'.format(layer_counter), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))]
            else:
                layers += [('maxpool{}'.format(layer_counter), nn.MaxPool2d(kernel_size=2, stride=2))]
            maxpool_counter += 1
            layer_counter += 1
            conv_counter = 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            layers += make_gated_layer(layer_counter, conv_counter, v, conv2d, batch_norm=batch_norm, no_gating=no_gating)
            conv_counter += 1
            in_channels = v

    vgg = nn.Sequential(OrderedDict(layers))
    for c in vgg.children():
        if isinstance(c, nn.Conv2d):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)
            elif init_method == 'kaiming-fan-out':
                nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity='relu')


    return vgg, in_channels


def make_gated_layer(layer_counter, conv_counter, out_channels, conv2d, batch_norm=False, no_gating=False, no_relu=False):
    layers = []
    if conv_counter is None:
        conv_counter_ = ''
    else:
        conv_counter_ = '-' + str(conv_counter)
    layers += [('conv{}{}'.format(layer_counter, conv_counter_), conv2d)]
    if batch_norm:
        layers += [('bn{}{}'.format(layer_counter, conv_counter_), nn.BatchNorm2d(out_channels))]
    if not no_gating:
        layers += [('gate{}{}'.format(layer_counter, conv_counter_), GateLayer(out_channels, out_channels, [1, -1, 1, 1]))]
    if not no_relu:
        layers += [('relu{}{}'.format(layer_counter, conv_counter_), nn.ReLU(inplace=True))]
    return layers

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M']
}


def vgg11(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 11-layer model (configuration "A")"""
    vgg, in_channels = make_layers(cfg['A'], bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg11(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg11_bn(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    vgg, in_channels = make_layers(cfg['A'], batch_norm=True, bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg11_bn(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg13(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 13-layer model (configuration "B")"""
    vgg, in_channels = make_layers(cfg['B'], bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg13(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg13_bn(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    vgg, in_channels = make_layers(cfg['B'], batch_norm=True, bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg13_bn(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg16(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 16-layer model (configuration "D")"""
    vgg, in_channels = make_layers(cfg['D'], bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg16_bn(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    vgg, in_channels = make_layers(cfg['D'], batch_norm=True, bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg19(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 19-layer model (configuration "E")"""
    vgg, in_channels = make_layers(cfg['E'], bias=bias, no_gating=no_gating, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg19(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels


def vgg19_bn(bias=False, no_gating=False, pretrained=False, init_method=False):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    vgg, in_channels = make_layers(cfg['E'], batch_norm=True, bias=bias, no_gating=no_gating)
    if pretrained:
        pretrained_state_dict = torchvision.models.vgg19_bn(pretrained=True).state_dict()
        loadPretrainedWeights(vgg, pretrained_state_dict)
    return vgg, in_channels

def loadPretrainedWeights(vgg, pretrained_state_dict):
    pretrained_param_names = list(pretrained_state_dict.keys())
    curr_dict = vgg.state_dict()
    curr_dict_keys = list(curr_dict.keys())

    for i in range(len(pretrained_param_names)):
        if pretrained_param_names[i].startswith('features.34'):
            break
        assert pretrained_param_names[i].split('.')[-1] == curr_dict_keys[i].split('.')[-1]
        curr_dict[curr_dict_keys[i]] = pretrained_state_dict[pretrained_param_names[i]]
    vgg.load_state_dict(curr_dict)