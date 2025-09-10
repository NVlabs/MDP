'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer
import os
import numpy as np


first_channel, channels, block_output = None, None, None


def set_channels(arch, layer_num_dict):
    if arch == 'resnet50':
        first_channel = layer_num_dict['module.conv1']
        channels = [[[layer_num_dict['module.layer1.0.conv1'], layer_num_dict['module.layer1.0.conv2']], [layer_num_dict['module.layer1.1.conv1'], layer_num_dict['module.layer1.1.conv2']], [layer_num_dict['module.layer1.2.conv1'], layer_num_dict['module.layer1.2.conv2']]],
                    [[layer_num_dict['module.layer2.0.conv1'], layer_num_dict['module.layer2.0.conv2']], [layer_num_dict['module.layer2.1.conv1'], layer_num_dict['module.layer2.1.conv2']], [layer_num_dict['module.layer2.2.conv1'], layer_num_dict['module.layer2.2.conv2']], [layer_num_dict['module.layer2.3.conv1'], layer_num_dict['module.layer2.3.conv2']]],
                    [[layer_num_dict['module.layer3.0.conv1'], layer_num_dict['module.layer3.0.conv2']], [layer_num_dict['module.layer3.1.conv1'], layer_num_dict['module.layer3.1.conv2']], [layer_num_dict['module.layer3.2.conv1'], layer_num_dict['module.layer3.2.conv2']], [layer_num_dict['module.layer3.3.conv1'], layer_num_dict['module.layer3.3.conv2']], [layer_num_dict['module.layer3.4.conv1'], layer_num_dict['module.layer3.4.conv2']], [layer_num_dict['module.layer3.5.conv1'], layer_num_dict['module.layer3.5.conv2']]],
                    [[layer_num_dict['module.layer4.0.conv1'], layer_num_dict['module.layer4.0.conv2']], [layer_num_dict['module.layer4.1.conv1'], layer_num_dict['module.layer4.1.conv2']], [layer_num_dict['module.layer4.2.conv1'], layer_num_dict['module.layer4.2.conv2']]]]
        block_output = [layer_num_dict['module.layer1.2.conv3'], layer_num_dict['module.layer2.3.conv3'], layer_num_dict['module.layer3.5.conv3'], layer_num_dict['module.layer4.2.conv3']]
    elif arch == 'resnet34':
        first_channel = layer_num_dict['module.conv1']
        channels = [[layer_num_dict['module.layer1.0.conv1'], layer_num_dict['module.layer1.1.conv1'], layer_num_dict['module.layer1.2.conv1']],
                    [layer_num_dict['module.layer2.0.conv1'], layer_num_dict['module.layer2.1.conv1'], layer_num_dict['module.layer2.2.conv1'], layer_num_dict['module.layer2.3.conv1']],
                    [layer_num_dict['module.layer3.0.conv1'], layer_num_dict['module.layer3.1.conv1'], layer_num_dict['module.layer3.2.conv1'], layer_num_dict['module.layer3.3.conv1'], layer_num_dict['module.layer3.4.conv1'], layer_num_dict['module.layer3.5.conv1']],
                    [layer_num_dict['module.layer4.0.conv1'], layer_num_dict['module.layer4.1.conv1'], layer_num_dict['module.layer4.2.conv1']]]
        block_output = [layer_num_dict['module.layer1.2.conv2'], layer_num_dict['module.layer2.3.conv2'], layer_num_dict['module.layer3.5.conv2'], layer_num_dict['module.layer4.2.conv2']]
    elif arch == 'resnet101':
        first_channel = layer_num_dict['module.conv1']
        channels = [[[layer_num_dict['module.layer1.0.conv1'], layer_num_dict['module.layer1.0.conv2']], [layer_num_dict['module.layer1.1.conv1'], layer_num_dict['module.layer1.1.conv2']], [layer_num_dict['module.layer1.2.conv1'], layer_num_dict['module.layer1.2.conv2']]],
                    [[layer_num_dict['module.layer2.0.conv1'], layer_num_dict['module.layer2.0.conv2']], [layer_num_dict['module.layer2.1.conv1'], layer_num_dict['module.layer2.1.conv2']], [layer_num_dict['module.layer2.2.conv1'], layer_num_dict['module.layer2.2.conv2']], [layer_num_dict['module.layer2.3.conv1'], layer_num_dict['module.layer2.3.conv2']]],
                    [[layer_num_dict['module.layer3.0.conv1'], layer_num_dict['module.layer3.0.conv2']], [layer_num_dict['module.layer3.1.conv1'], layer_num_dict['module.layer3.1.conv2']], [layer_num_dict['module.layer3.2.conv1'], layer_num_dict['module.layer3.2.conv2']], [layer_num_dict['module.layer3.3.conv1'], layer_num_dict['module.layer3.3.conv2']], [layer_num_dict['module.layer3.4.conv1'], layer_num_dict['module.layer3.4.conv2']], [layer_num_dict['module.layer3.5.conv1'], layer_num_dict['module.layer3.5.conv2']],
                     [layer_num_dict['module.layer3.6.conv1'], layer_num_dict['module.layer3.6.conv2']], [layer_num_dict['module.layer3.7.conv1'], layer_num_dict['module.layer3.7.conv2']], [layer_num_dict['module.layer3.8.conv1'], layer_num_dict['module.layer3.8.conv2']], [layer_num_dict['module.layer3.9.conv1'], layer_num_dict['module.layer3.9.conv2']], [layer_num_dict['module.layer3.10.conv1'], layer_num_dict['module.layer3.10.conv2']], [layer_num_dict['module.layer3.11.conv1'], layer_num_dict['module.layer3.11.conv2']],
                     [layer_num_dict['module.layer3.12.conv1'], layer_num_dict['module.layer3.12.conv2']], [layer_num_dict['module.layer3.13.conv1'], layer_num_dict['module.layer3.13.conv2']], [layer_num_dict['module.layer3.14.conv1'], layer_num_dict['module.layer3.14.conv2']], [layer_num_dict['module.layer3.15.conv1'], layer_num_dict['module.layer3.15.conv2']], [layer_num_dict['module.layer3.16.conv1'], layer_num_dict['module.layer3.16.conv2']], [layer_num_dict['module.layer3.17.conv1'], layer_num_dict['module.layer3.17.conv2']],
                     [layer_num_dict['module.layer3.18.conv1'], layer_num_dict['module.layer3.18.conv2']], [layer_num_dict['module.layer3.19.conv1'], layer_num_dict['module.layer3.19.conv2']], [layer_num_dict['module.layer3.20.conv1'], layer_num_dict['module.layer3.20.conv2']], [layer_num_dict['module.layer3.21.conv1'], layer_num_dict['module.layer3.21.conv2']], [layer_num_dict['module.layer3.22.conv1'], layer_num_dict['module.layer3.22.conv2']]],
                    [[layer_num_dict['module.layer4.0.conv1'], layer_num_dict['module.layer4.0.conv2']], [layer_num_dict['module.layer4.1.conv1'], layer_num_dict['module.layer4.1.conv2']], [layer_num_dict['module.layer4.2.conv1'], layer_num_dict['module.layer4.2.conv2']]]]
        block_output = [layer_num_dict['module.layer1.2.conv3'], layer_num_dict['module.layer2.3.conv3'], layer_num_dict['module.layer3.5.conv3'], layer_num_dict['module.layer4.2.conv3']]
    else:
        raise NotImplementedError

    return first_channel, channels, block_output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, layer_index, block_index, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, gate_layer=None):
        super(BasicBlock, self).__init__()
        self.add_gate = False if gate_layer is None else True
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if channels[layer_index][block_index] == 0:
            self.conv1 = None
        else:
            self.conv1 = conv3x3(inplanes, channels[layer_index][block_index], stride)
            self.bn1 = norm_layer(channels[layer_index][block_index])
            self.gate1 = GateLayer(channels[layer_index][block_index], channels[layer_index][block_index], [1, -1, 1, 1]) if self.add_gate else None
        self.relu = nn.ReLU(inplace=True)
        if self.conv1 is None:
            self.conv2 = None
            self.add_bias = nn.Parameter(torch.zeros(1, block_output[layer_index], 1, 1), requires_grad=False)
        else:
            self.conv2 = conv3x3(channels[layer_index][block_index], block_output[layer_index])
            self.bn2 = norm_layer(block_output[layer_index])
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.gate_layer = gate_layer

    def forward(self, x):
        identity = x

        if self.conv1 is not None:
            out = self.conv1(x)
            out = self.bn1(out)
            if self.add_gate:
                out = self.gate1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        else:
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.add_bias + identity

        if self.gate_layer:
            out = self.gate_layer(out)
        if self.conv1 is not None:
            out = self.relu(out)
        else:
            out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, layer_index, block_index, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, gate_layer=None):
        super(Bottleneck, self).__init__()
        self.add_gate = False if gate_layer is None else True
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = block_output[layer_index]
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if channels[layer_index][block_index][0] == 0 or channels[layer_index][block_index][1] == 0:
            self.conv1 = None
        else:
            self.conv1 = conv1x1(inplanes, channels[layer_index][block_index][0])
            self.bn1 = norm_layer(channels[layer_index][block_index][0])
            self.gate1 = GateLayer(channels[layer_index][block_index][0], channels[layer_index][block_index][0], [1, -1, 1, 1]) if self.add_gate else None
        if channels[layer_index][block_index][0] == 0 or channels[layer_index][block_index][1] == 0:
            self.conv2 = None
        else:
            self.conv2 = conv3x3(channels[layer_index][block_index][0], channels[layer_index][block_index][1], stride, groups, dilation)
            self.bn2 = norm_layer(channels[layer_index][block_index][1])
            self.gate2 = GateLayer(channels[layer_index][block_index][1], channels[layer_index][block_index][1], [1, -1, 1, 1]) if self.add_gate else None
        if self.conv1 is None or self.conv2 is None:
            self.conv3 = None
            self.add_bias = nn.Parameter(torch.zeros(1, width, 1, 1), requires_grad=False)
        else:
            self.conv3 = conv1x1(channels[layer_index][block_index][1], width)
            self.bn3 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.gate_layer = gate_layer

    def forward(self, x):
        identity = x

        if self.conv1 is not None and self.conv2 is not None:
            out = self.conv1(x)
            out = self.bn1(out)
            if self.add_gate:
                out = self.gate1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if self.add_gate:
                out = self.gate2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        else:
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.add_bias + identity

        if self.gate_layer:
            out = self.gate_layer(out)
        if self.conv1 is not None and self.conv2 is not None:
            out = self.relu(out)
        else:
            out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, arch, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, add_gate=False, small_kernel=False):
        super(ResNet, self).__init__()
        self.layers_have_downsample = []
        if arch == 'resnet34':
            self.layers_have_downsample = [1, 2, 3]
        elif arch == 'resnet50':
            self.layers_have_downsample = [0, 1, 2, 3]
        elif arch == 'resnet101':
            self.layers_have_downsample = [0, 1, 2, 3]

        self.add_gate = add_gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = first_channel
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if not small_kernel: # conv as in stardard imagenet resnet
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.gate1 = GateLayer(self.inplanes, self.inplanes, [1, -1, 1, 1]) if add_gate else None
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.gate_skip64 = GateLayer(block_output[0], block_output[0], [1, -1, 1, 1]) if add_gate else None
        self.gate_skip128 = GateLayer(block_output[1], block_output[1], [1, -1, 1, 1]) if add_gate else None
        self.gate_skip256 = GateLayer(block_output[2], block_output[2], [1, -1, 1, 1]) if add_gate else None
        self.gate_skip512 = GateLayer(block_output[3], block_output[3], [1, -1, 1, 1]) if add_gate else None
        self.layer1 = self._make_layer(block, 0, layers[0], gate_layer=self.gate_skip64)
        self.layer2 = self._make_layer(block, 1, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], gate_layer=self.gate_skip128)
        self.layer3 = self._make_layer(block, 2, layers[2], stride=2, 
                                       dilate=replace_stride_with_dilation[1], gate_layer=self.gate_skip256)
        self.layer4 = self._make_layer(block, 3, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], gate_layer=self.gate_skip512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_output[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, layer_index, blocks, stride=1, dilate=False, gate_layer=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if layer_index in self.layers_have_downsample:
            downsample = nn.Sequential(
                conv1x1(self.inplanes,  # residual_channels[block_index - 1] if block_index > 0 else first_channel
                        block_output[layer_index], stride),
                norm_layer(block_output[layer_index]),
            )

        layers = []
        layers.append(block(self.inplanes, layer_index, 0, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, gate_layer=gate_layer))
        self.inplanes = block_output[layer_index]
        for i in range(1, blocks):
            layers.append(block(self.inplanes, layer_index, i, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, gate_layer=gate_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        if self.add_gate:
            x = self.gate1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# def ResNet18(num_class=10, add_gate=False):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_class, add_gate=add_gate)


def ResNet34(arch, layer_num_dict, num_class=10, add_gate=False, small_kernel=False):
    global first_channel, channels, block_output
    first_channel, channels, block_output = set_channels(arch, layer_num_dict)
    return ResNet(arch, BasicBlock, [3, 4, 6, 3], num_class, add_gate=add_gate, small_kernel=small_kernel)


def ResNet50(arch, layer_num_dict, num_class=10, add_gate=False, small_kernel=False):
    global first_channel, channels, block_output
    first_channel, channels, block_output = set_channels(arch, layer_num_dict)
    return ResNet(arch, Bottleneck, [3, 4, 6, 3], num_class, add_gate=add_gate, small_kernel=small_kernel)

def ResNet101(arch, layer_num_dict, num_class=10, add_gate=False, small_kernel=False):
    global first_channel, channels, block_output
    first_channel, channels, block_output = set_channels(arch, layer_num_dict)
    return ResNet(arch, Bottleneck, [3, 4, 23, 3], num_class, add_gate=add_gate, small_kernel=small_kernel)


# def ResNet101(num_class=10, add_gate=False):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_class, add_gate=add_gate)
#
#
# def ResNet152(num_class=10, add_gate=False):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_class, add_gate=add_gate)
