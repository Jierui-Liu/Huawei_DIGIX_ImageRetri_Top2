'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-07-31 01:07:19
'''
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import warnings


# __all__ = ['ResNet_IBN', 'resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',
#            'resnet18_ibn_b', 'resnet34_ibn_b', 'resnet50_ibn_b', 'resnet101_ibn_b', 'resnet152_ibn_b']

model_urls = {
    'resnet18_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'resnet34_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet18_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth',
    'resnet34_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
    'densenet169_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet169_ibn_a-9f32c161.pth',
    'densenet121_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet121_ibn_a-e4af5cc1.pth',

}



class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net" 
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * (1-ratio))
        self.BN = nn.BatchNorm2d(self.half)
        self.IN = nn.InstanceNorm2d(planes - self.half, affine=True)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.BN(split[0].contiguous())
        out2 = self.IN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def densenet121_ibn_a(pretrained=False, **kwargs):
    r"""Densenet-121-IBN-a model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_IBN(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['densenet121_ibn_a']))
    model = nn.Sequential(*list(model.children())[:1])
    return model


def densenet169_ibn_a(pretrained=False, **kwargs):
    r"""Densenet-169-IBN-a model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_IBN(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['densenet169_ibn_a'],map_location='cpu'))
        model.load_state_dict(torch.load('./densenet169_ibn_a-9f32c161.pth',map_location='cpu'))
    model = nn.Sequential(*list(model.children())[:1])
    return model


def densenet201_ibn_a(pretrained=False, **kwargs):
    r"""Densenet-201-IBN-a model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_IBN(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for Densenet-201-IBN-a!")
    model = nn.Sequential(*list(model.children())[:1])
    return model


def densenet161_ibn_a(pretrained=False, **kwargs):
    r"""Densenet-161-IBN-a model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_IBN(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for Densenet-161-IBN-a!")
    model = nn.Sequential(*list(model.children())[:1])
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, ibn):
        super(_DenseLayer, self).__init__()
        if ibn:
            self.add_module('norm1', IBN(num_input_features, 0.4)),
        else:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, ibn):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            if ibn and i % 3 == 0:
                layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, True)
            else:
                layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, False)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_IBN(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet_IBN, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            ibn = True
            if i >= 3:
                ibn = False
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, ibn=ibn)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out







class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# def resnet18_ibn_a(pretrained=False, **kwargs):
#     """Constructs a ResNet-18-IBN-a model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet_IBN(block=BasicBlock_IBN,
#                        layers=[2, 2, 2, 2],
#                        ibn_cfg=('a', 'a', 'a', None),
#                        **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet18_ibn_a']))
#     return model


# def resnet34_ibn_a(pretrained=False, **kwargs):
#     """Constructs a ResNet-34-IBN-a model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet_IBN(block=BasicBlock_IBN,
#                        layers=[3, 4, 6, 3],
#                        ibn_cfg=('a', 'a', 'a', None),
#                        **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_a']))
#     return model


def resnet50_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a']))
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a']))
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-a!")
    model = nn.Sequential(*list(model.children())[:-2])
    return model


# def resnet18_ibn_b(pretrained=False, **kwargs):
#     """Constructs a ResNet-18-IBN-b model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet_IBN(block=BasicBlock_IBN,
#                        layers=[2, 2, 2, 2],
#                        ibn_cfg=('b', 'b', None, None),
#                        **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet18_ibn_b']))
#     return model


# def resnet34_ibn_b(pretrained=False, **kwargs):
#     """Constructs a ResNet-34-IBN-b model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet_IBN(block=BasicBlock_IBN,
#                        layers=[3, 4, 6, 3],
#                        ibn_cfg=('b', 'b', None, None),
#                        **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_b']))
#     return model


def resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_b']))
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_b']))
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet152_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-b!")
    model = nn.Sequential(*list(model.children())[:-2])
    return model


if __name__ == "__main__":
    # model = resnet101_ibn_a()
    model = densenet161_ibn_a()
    # model = nn.Sequential(*list(model.children())[:-2])
    print(model)