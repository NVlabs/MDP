from models.vgg import *
from models.resnet import *
from models.resnet_collapsible import ResNet50 as CollapsibleResNet50
from models.resnet_collapsible2 import ResNet50 as CollapsibleResNet50V2
from models.mobilenet import mobilenet
from models.resnet_nobn import ResNet50 as ResNet50_nobn


def get_model(model_name, dataset_name, enable_bias, gate=False, collapse=False, collapse2=False):
    if model_name == 'vgg11':
        model = vgg11(enable_bias, dataset_name)
    elif model_name == 'vgg11_bn':
        model = vgg11_bn(enable_bias, dataset_name)
    elif model_name == 'vgg13':
        model = vgg13(enable_bias, dataset_name)
    elif model_name == 'vgg13_bn':
        model = vgg13_bn(enable_bias, dataset_name)
    elif model_name == 'vgg16':
        model = vgg16(enable_bias, dataset_name)
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(enable_bias, dataset_name)
    elif model_name == 'vgg19':
        model = vgg19(enable_bias, dataset_name)
    elif model_name == 'vgg19_bn':
        model = vgg19_bn(enable_bias, dataset_name)
    elif model_name == 'mobilenet':
        model = mobilenet()
    else:
        num_class = 1000
        small_kernel = False
        if dataset_name == 'CIFAR10':
            num_class = 10
            small_kernel = True
        elif dataset_name == 'ImageNet':
            num_class = 1000
            small_kernel = False
        else:
            NotImplementedError('Network for dataset {} is not implemented.'.format(dataset_name))
        if model_name == 'resnet18':
            model = ResNet18(num_class, gate, small_kernel)
        elif model_name == 'resnet34':
            model = ResNet34(num_class, gate, small_kernel)
        elif model_name == 'resnet50' and not collapse and not collapse2:
            model = ResNet50(num_class, gate, small_kernel)
        elif model_name == 'resnet50' and collapse2 and not collapse:
            model = CollapsibleResNet50V2(num_class, gate, small_kernel)
        elif model_name == 'resnet50' and collapse and not collapse2:
            model = CollapsibleResNet50(num_class, gate, small_kernel)
        elif model_name == 'resnet101':
            model = ResNet101(num_class, gate, small_kernel)
        elif model_name == 'resnet152':
            model = ResNet152(num_class, gate, small_kernel)
        elif model_name == 'resnet50_nobn':
            model = ResNet50_nobn(num_class)
        else:
            NotImplementedError('Pruning for architecture {} is not implemented'.format(model_name))

    return model
