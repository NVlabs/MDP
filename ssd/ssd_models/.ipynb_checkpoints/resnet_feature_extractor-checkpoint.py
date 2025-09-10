'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, gate_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, gate_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, init_method='xavier'):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], gate_layer=None)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], gate_layer=None)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1], gate_layer=None)
        self.out_channels = 256 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias:
                        nn.init.constant_(m.bias, 0.)
                elif init_method == 'kaiming-fan-out':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, gate_layer=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, gate_layer=gate_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, gate_layer=gate_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def ResNet18(num_class=21, pretrained=False, init_method='xavier'):
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_class, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.resnet18(pretrained=True).state_dict()
        loadPretrainedWeights(resnet, pretrained_state_dict)
    return resnet, resnet.out_channels


def ResNet34(num_class=21, pretrained=False, init_method='xavier', checkpoint=None):
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_class, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.resnet34(pretrained=True).state_dict()
        loadPretrainedWeights(resnet, pretrained_state_dict)
    if checkpoint is not None:
        checkpoint_state_dict = torch.load(checkpoint)
        loadPretrainedWeights(resnet, checkpoint_state_dict)
    return resnet, resnet.out_channels


def ResNet50(num_class=21, pretrained=False, init_method='xavier', checkpoint=None):
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_class, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        loadPretrainedWeights(resnet, pretrained_state_dict)
    if checkpoint is not None:
        checkpoint_state_dict = torch.load(checkpoint)
        loadPretrainedWeights(resnet, checkpoint_state_dict)
    return resnet, resnet.out_channels

def ResNet101(num_class=21, pretrained=False, init_method='xavier'):
    resnet = ResNet(Bottleneck, [3, 4, 23, 3], num_class, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.resnet101(pretrained=True).state_dict()
        loadPretrainedWeights(resnet, pretrained_state_dict)
    return resnet, resnet.out_channels

def ResNet152(num_class=21, pretrained=False, init_method='xavier'):
    resnet = ResNet(Bottleneck, [3, 8, 36, 3], num_class, init_method=init_method)
    if pretrained:
        pretrained_state_dict = torchvision.models.resnet152(pretrained=True).state_dict()
        loadPretrainedWeights(resnet, pretrained_state_dict)
    return resnet, resnet.out_channels


def loadPretrainedWeights(resnet, pretrained_state_dict):
    pretrained_param_names = list(pretrained_state_dict['model_state_dict'].keys())
    curr_dict = resnet.state_dict()
    for i in range(len(pretrained_param_names)):
        curr_name = pretrained_param_names[i]
        if curr_name.startswith('module.f_0.'):
            curr_name = curr_name[11:]
        if curr_name.startswith('layer4'):
            break
        if curr_name.startswith('gate_skip512') or curr_name.startswith('module.features_') or curr_name.startswith('module.predictor.'):
            continue
        #assert curr_name in curr_dict
        curr_dict[curr_name] = pretrained_state_dict['model_state_dict'][pretrained_param_names[i]]

    resnet.load_state_dict(curr_dict)
