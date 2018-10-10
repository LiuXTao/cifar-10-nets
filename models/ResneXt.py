import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
论文：Aggregated Residual Transformations for Deep Neural Networks
提出 ResNeXt 的主要原因在于：传统的要提高模型的准确率，都是加深或加宽网络，但是随着超参数数量的增加（比如channels数，filter size等等），
网络设计的难度和计算开销也会增加。因此本文提出的ResNeXt 结构可以在不增加参数复杂度的前提下提高准确率，同时还减少了超参数的数量
（得益于子模块的拓扑结构一样，后面会讲）。


'''

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Bottleneck, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, group_width, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(group_width, Bottleneck.expansion * group_width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(group_width)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != Bottleneck.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, Bottleneck.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * group_width)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __int__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__int__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(1, num_blocks[0], Bottleneck)
        self.layer2 = self._make_layer(2, num_blocks[1], Bottleneck)
        self.layer3 = self._make_layer(2, num_blocks[2], Bottleneck)
        self.avg_pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(cardinality*bottleneck_width* math.pow(2, 3), num_classes)


    def _make_layer(self, stride, num_blocks, Block):
        strides = [stride] + [1] * (num_blocks - 1)  # 拼凑一个strides数组
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)

def ResNeXt29_4x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64)

def ResNeXt29_8x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64)

def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4)

def test_resnext():
    net = ResNeXt29_2x64d()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test_resnext()
