import torch
from torch import nn
import torch.nn.functional as F
from ssd_models.vgg_feature_extractor import *
from ssd_models.resnet_feature_extractor import *
from collections import OrderedDict
from utils.ssd_utils import *
from layers.gate_layer import GateLayer


def get_feat_extractor(model_name, dataset_name, enable_bias, no_gating=False, pretrained=False, init_method='xavier', checkpoint=None):
    if model_name == 'vgg11':
        feat_extractor, out_channels = vgg11(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg11_bn':
        feat_extractor, out_channels = vgg11_bn(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg13':
        feat_extractor, out_channels = vgg13(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg13_bn':
        feat_extractor, out_channels = vgg13_bn(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg16':
        feat_extractor, out_channels = vgg16(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg16_bn':
        feat_extractor, out_channels = vgg16_bn(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg19':
        feat_extractor, out_channels = vgg19(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    elif model_name == 'vgg19_bn':
        feat_extractor, out_channels = vgg19_bn(enable_bias, no_gating=no_gating, pretrained=pretrained, init_method=init_method)
    else:
        if dataset_name == 'VOC':
            num_class = 21
        else:
            NotImplementedError('Network for dataset {} is not implemented.'.format(dataset_name))
        if model_name == 'resnet18':
            feat_extractor, out_channels = ResNet18(num_class, pretrained=pretrained, init_method=init_method)
        elif model_name == 'resnet34':
            feat_extractor, out_channels = ResNet34(num_class, pretrained=pretrained, init_method=init_method, checkpoint=checkpoint)
        elif model_name == 'resnet50':
            feat_extractor, out_channels = ResNet50(num_class, pretrained=pretrained, init_method=init_method, checkpoint=checkpoint)
        elif model_name == 'resnet101':
            feat_extractor, out_channels = ResNet101(num_class, pretrained=pretrained, init_method=init_method)
        elif model_name == 'resnet152':
            feat_extractor, out_channels = ResNet152(num_class, pretrained=pretrained, init_method=init_method)
        else:
            NotImplementedError('Pruning for architecture {} is not implemented'.format(model_name))

    return feat_extractor, out_channels


# This is from the original SSD paper. However, using the skim version is lighter and does not lose accuracy.
def get_ssd_additional_layers(enable_bias, in_channels, bias=False, batch_norm=False, no_gating=False, is_SSD300=True, init_method='xavier'):
    all_layers = []
    # Layers until conv8_2
    layers_1 = []
    layers_1 += [('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))]
    layers_1 += make_gated_layer(layer_counter=5, conv_counter=1, out_channels=512,
                                 conv2d=nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_1 += make_gated_layer(layer_counter=5, conv_counter=2, out_channels=512,
                                 conv2d=nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_1 += make_gated_layer(layer_counter=5, conv_counter=3, out_channels=512,
                                 conv2d=nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_1 += [('maxpool5', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))]
    layers_1 += make_gated_layer(layer_counter=6, conv_counter=None, out_channels=1024,
                                 conv2d=nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
                                 #conv2d=nn.Conv2d(512, 1024, kernel_size=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_1 += make_gated_layer(layer_counter=7, conv_counter=None, out_channels=1024,
                                 conv2d=nn.Conv2d(1024, 1024, kernel_size=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_1)))

    layers_2 = []
    layers_2 += make_gated_layer(layer_counter=8, conv_counter=1, out_channels=256,
                                 conv2d=nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_2 += make_gated_layer(layer_counter=8, conv_counter=2, out_channels=512,
                                 conv2d=nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_2)))
    
    # Layers until conv9_2
    layers_3 = []
    layers_3 += make_gated_layer(layer_counter=9, conv_counter=1, out_channels=128,
                                 conv2d=nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_3 += make_gated_layer(layer_counter=9, conv_counter=2, out_channels=256,
                                 conv2d=nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_3)))

    # Layers until conv10_2
    layers_4 = []
    layers_4 += make_gated_layer(layer_counter=10, conv_counter=1, out_channels=128,
                                 conv2d=nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    if is_SSD300:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=0, bias=bias)
    else:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias)
    layers_4 += make_gated_layer(layer_counter=10, conv_counter=2, out_channels=256, conv2d=conv2d, batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_4)))

    # Layers until conv11_2
    layers_5 = []
    layers_5 += make_gated_layer(layer_counter=11, conv_counter=1, out_channels=128,
                                 conv2d=nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    if is_SSD300:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=0, bias=bias)
    else:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias)
    layers_5 += make_gated_layer(layer_counter=11, conv_counter=2, out_channels=256, conv2d=conv2d, batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_5)))

    if not is_SSD300:
        layers_6 = []
        layers_6 += make_gated_layer(layer_counter=12, conv_counter=1, out_channels=128,
                                     conv2d=nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
        layers_6 += make_gated_layer(layer_counter=12, conv_counter=2, out_channels=256,
                                     conv2d=nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
        all_layers.append(nn.Sequential(OrderedDict(layers_6)))
    else:
        all_layers.append(None)

    # Initialization
    for layer in all_layers:
        if layer:
            for c in layer.children():
                if isinstance(c, nn.Conv2d):
                    if init_method == 'xavier':
                        nn.init.xavier_uniform_(c.weight)
                        if c.bias is not None:
                            nn.init.constant_(c.bias, 0.)
                    elif init_method == 'kaiming-fan-out':
                        nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity='relu')
    return all_layers


def get_ssd_additional_layers_skim(enable_bias, in_channels, bias=False, batch_norm=False, no_gating=False, is_SSD300=True, init_method='xavier'):
    all_layers = []
    # Layers until conv8_2
    layers_1 = []
    layers_1 += make_gated_layer(layer_counter=5, conv_counter=1, out_channels=512,
                                 conv2d=nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_1 += make_gated_layer(layer_counter=5, conv_counter=2, out_channels=512,
                                 conv2d=nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_1)))

    layers_2 = []
    layers_2 += make_gated_layer(layer_counter=8, conv_counter=1, out_channels=256,
                                 conv2d=nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_2 += make_gated_layer(layer_counter=8, conv_counter=2, out_channels=512,
                                 conv2d=nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_2)))
    
    # Layers until conv9_2
    layers_3 = []
    layers_3 += make_gated_layer(layer_counter=9, conv_counter=1, out_channels=128,
                                 conv2d=nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    layers_3 += make_gated_layer(layer_counter=9, conv_counter=2, out_channels=256,
                                 conv2d=nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_3)))

    # Layers until conv10_2
    layers_4 = []
    layers_4 += make_gated_layer(layer_counter=10, conv_counter=1, out_channels=128,
                                 conv2d=nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    if is_SSD300:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=0, bias=bias)
    else:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias)
    layers_4 += make_gated_layer(layer_counter=10, conv_counter=2, out_channels=256, conv2d=conv2d, batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_4)))

    # Layers until conv11_2
    layers_5 = []
    layers_5 += make_gated_layer(layer_counter=11, conv_counter=1, out_channels=128,
                                 conv2d=nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
    if is_SSD300:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=0, bias=bias)
    else:
        conv2d = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias)
    layers_5 += make_gated_layer(layer_counter=11, conv_counter=2, out_channels=256, conv2d=conv2d, batch_norm=batch_norm, no_gating=no_gating)
    all_layers.append(nn.Sequential(OrderedDict(layers_5)))

    if not is_SSD300:
        layers_6 = []
        layers_6 += make_gated_layer(layer_counter=12, conv_counter=1, out_channels=128,
                                     conv2d=nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
        layers_6 += make_gated_layer(layer_counter=12, conv_counter=2, out_channels=256,
                                     conv2d=nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=1, bias=bias), batch_norm=batch_norm, no_gating=no_gating)
        all_layers.append(nn.Sequential(OrderedDict(layers_6)))
    else:
        all_layers.append(None)

    # Initialization
    for layer in all_layers:
        if layer:
            for c in layer.children():
                if isinstance(c, nn.Conv2d):
                    if init_method == 'xavier':
                        nn.init.xavier_uniform_(c.weight)
                        if c.bias is not None:
                            nn.init.constant_(c.bias, 0.)
                    elif init_method == 'kaiming-fan-out':
                        nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity='relu')
    return all_layers


class SSDPredictions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, conv4_out_channels, is_SSD300=True, init_method='xavier'):
        """
        :param n_classes: number of different types of objects
        """
        super(SSDPredictions, self).__init__()

        self.n_classes = n_classes
        self.is_SSD300 = is_SSD300
        self.init_method = init_method

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        if not self.is_SSD300:
            n_boxes['conv10_2'] = 6
            n_boxes['conv12_2'] = 4
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        channels = n_boxes['conv4_3'] * 4
        self.loc_conv4_3 = nn.Conv2d(conv4_out_channels, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv7'] * 4
        self.loc_conv7 = nn.Conv2d(1024, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv8_2'] * 4
        self.loc_conv8_2 = nn.Conv2d(512, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv9_2'] * 4
        self.loc_conv9_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv10_2'] * 4
        self.loc_conv10_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv11_2'] * 4
        self.loc_conv11_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        channels = n_boxes['conv4_3'] * n_classes
        self.cl_conv4_3 = nn.Conv2d(conv4_out_channels, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv7'] * n_classes
        self.cl_conv7 = nn.Conv2d(1024, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv8_2'] * n_classes
        self.cl_conv8_2 = nn.Conv2d(512, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv9_2'] * n_classes
        self.cl_conv9_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv10_2'] * n_classes
        self.cl_conv10_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)
        channels = n_boxes['conv11_2'] * n_classes
        self.cl_conv11_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)
        
        if not self.is_SSD300:
            channels = n_boxes['conv12_2'] * 4
            self.loc_conv12_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)
            channels = n_boxes['conv12_2'] * n_classes
            self.cl_conv12_2 = nn.Conv2d(256, channels, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                if self.init_method == 'xavier':
                    nn.init.xavier_uniform_(c.weight)
                    if c.bias is not None:
                        nn.init.constant_(c.bias, 0.)
                elif self.init_method == 'kaiming-fan-out':
                    nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats, conv12_2_feats=None):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38) / ((N, 512, 64, 64) for SSD512)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19) / (N, 1024, 32, 32)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10) / (N, 512, 16, 16)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5) / (N, 256, 8, 8)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3) / (N, 256, 4, 4)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1) / (N, 256, 2, 2)
        :param conv12_2_feats: conv12_2 feature map, a tensor of dimensions (N/A) / (N, 256, 1, 1)
        :return: 8732 or 24564 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        if not self.is_SSD300:
            l_conv12_2 = self.loc_conv12_2(conv12_2_feats)  # (N, 16, 1, 1)
            l_conv12_2 = l_conv12_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
            l_conv12_2 = l_conv12_2.view(batch_size, -1, 4)  # (N, 4, 4)

            c_conv12_2 = self.cl_conv12_2(conv12_2_feats)  # (N, 4 * n_classes, 1, 1)
            c_conv12_2 = c_conv12_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
            c_conv12_2 = c_conv12_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)
            locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2, l_conv12_2], dim=1)  # (N, 24564, 4)
            classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2, c_conv12_2],
                                       dim=1)  # (N, 24564, n_classes)
        else:
            # A total of 8732 boxes
            # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
            locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
            classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                       dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD(nn.Module):
    def __init__(self, model='SSD300', backbone='resnet34', n_classes=21, pretrained=False, bias=False, batch_norm=False, no_gating=False, init_method='xavier', checkpoint=None):
        super(SSD, self).__init__()
        self.is_SSD300 = model == 'SSD300'
        self.n_classes = n_classes

        if backbone.startswith('vgg') and not backbone.endswith('_bn') and batch_norm:
            backbone_ = backbone + '_bn'
        else:
            backbone_ = backbone
        self.f_0, out_channels = get_feat_extractor(model_name=backbone_, dataset_name='VOC', enable_bias=True, no_gating=no_gating, \
                                                    pretrained=pretrained, init_method=init_method, checkpoint=checkpoint)
        # !important
        # out_channels = 448
        out_channels = 1024
        self.features_1, self.features_2, self.features_3, self.features_4, self.features_5, self.features_6 = \
            get_ssd_additional_layers(enable_bias=False, in_channels=out_channels, bias=bias,  \
                                      batch_norm=batch_norm, no_gating=no_gating, is_SSD300=self.is_SSD300, init_method=init_method)

        self.predictor = SSDPredictions(n_classes, conv4_out_channels=out_channels, is_SSD300=self.is_SSD300, init_method=init_method)

        # Prior boxes
        self.priors_cxcy = create_VOC_prior_boxes(is_SSD300=self.is_SSD300)

    def forward(self, image):
        feats_0 = self.f_0(image)  # (N, 512, 38, 38) or (N, 512, 64, 64)
        # print(feats_0.shape)
        feats_1 = self.features_1(feats_0)  # (N, 1024, 19, 19) or (N, 1024, 32, 32)
        feats_2 = self.features_2(feats_1)  # (N, 512, 10, 10) or (N, 512, 16, 16)
        feats_3 = self.features_3(feats_2)  # (N, 256, 5, 5) or (N, 256, 8, 8)
        feats_4 = self.features_4(feats_3)  # (N, 256, 3, 3) or (N, 256, 4, 4)
        feats_5 = self.features_5(feats_4)  # (N, 256, 1, 1) or (N, 256, 2, 2)
        if not self.is_SSD300:
            feats_6 = self.features_6(feats_5)  # (N, 256, 1, 1)
        else:
            feats_6 = None

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.predictor(feats_0, feats_1, feats_2, feats_3, feats_4, feats_5, feats_6)

        return locs, classes_scores  # (N, 8732, 4), (N, 8732, n_classes) or (N, 24564, 4), (N, 24564, n_classes)
