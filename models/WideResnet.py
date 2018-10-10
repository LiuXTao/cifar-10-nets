import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
motivation：ResNet的跳连接，导致了只有少量的残差块学到了有用信息，或者大部分残差块只能提供少量的信息。
            于是作者探索一种新的网络WideResNet（在ResNet的基础上减小深度，增加宽度）。

网络结构：在ResNetv2的基础上改进，增大每个残差块中的卷积核数量。如下两个图所示。其中B(3,3)表示一个两个3x3卷积，k表示一个宽度因子，
当k为1时卷积核个数和ResNetv2相等，k越大网络越宽。另外WRN在卷积层之间加入dropout（下一个卷积层之前的bn和relu之后），
如下第一个图的图(d)所示（在ResNetv2中把dropout放在恒等映射中实验发现效果不好于是放弃了dropout）。用WRN-n-k来表示一个网络，n表示卷积层的总数，k表示宽度因子。

探究宽度对于网络性能的影响.首先我们说明一下什么是宽度.对于卷积层来说,宽度是指输出维度,
如ResNet50的第一个卷积层参数为(64,3,7,7)，宽度即输出维度也就是64.而对于一个网络来说,宽度则是指所有参数层的总体输出维度数.

'''

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super(BasicBlock, self).__init__()

        self.pre1 = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.pre2 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.dropout = dropout
        self.equalInOut = (in_planes==out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.pre1(x)
        else:
            out = self.pre1(x)
        out = self.pre2(self.conv1(out if self.equalInOut else x))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WideResNet(nn.Module):
    def __init__(self, block, depth, num_classes=10, widen_factor=1, dropout=0.0):
        super(WideResNet, self).__init__()
        self.block = block
        n_channels = [16, 16*widen_factor, 32 *widen_factor, 48*widen_factor]
        self.nchannel = n_channels[3]
        assert ((depth-4) % 6 == 0)
        n = int((depth-4) / 6)
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_layer(n, n_channels[0], n_channels[1], 1, dropout)
        self.block2 = self._make_layer(n, n_channels[1], n_channels[2], 2, dropout)
        self.block3 = self._make_layer(n, n_channels[2], n_channels[3], 3, dropout)

        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(n_channels[3]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8)
        )
        self.linear = self.Linear(n_channels[3], num_classes)


    def _make_layer(self, nb_layer, in_planes, out_planes,  stride, dropout=0.0):
        layer = []
        layer.append(self.block(in_planes, out_planes, stride, dropout))
        for i in range(1, nb_layer):
            layer.append(self.block(out_planes, out_planes, 1, dropout))

        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avg_pool(out)
        out = out.view(-1, self.nchannel)
        out = self.linear(out)
        return out
