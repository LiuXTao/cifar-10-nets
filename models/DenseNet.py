import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
network:就是每一层的输入来自前面所有层的输出。

DenseNet的一个优点是网络更窄，参数更少，很大一部分原因得益于这种dense block的设计，后面有提到在dense block中每个卷积层的输出feature map的数量都很小（小于100），
而不是像其他网络一样动不动就几百上千的宽度。同时这种连接方式使得特征和梯度的传递更加有效，网络也就更加容易训练。原文的一句话非常喜欢：
Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision.
直接解释了为什么这个网络的效果会很好。前面提到过梯度消失问题在网络深度越深的时候越容易出现，原因就是输入信息和梯度信息在很多层之间传递导致的，
而现在这种dense connection相当于每一层都直接连接input和loss，因此就可以减轻梯度消失现象，这样更深网络不是问题。另外作者还观察到这种dense connection有正则化的效果，
因此对于过拟合有一定的抑制作用，博主认为是因为参数减少了（后面会介绍为什么参数会减少），所以过拟合现象减轻。

bottleneck layer  && transition layer


该文章提出的DenseNet核心思想在于建立了不同层之间的连接关系，充分利用了feature，进一步减轻了梯度消失问题，加深网络不是问题，而且训练效果非常好。
另外，利用bottleneck layer，transition layer以及较小的growth rate使得网络变窄，参数减少，有效抑制了过拟合，同时计算量也减少了。DenseNet优点很多，
而且在和ResNet的对比中优势还是非常明显的。


'''

class bottleneck_block(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(bottleneck_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, growth_rate*4, kernel_size=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate*4, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)

        return torch.cat([res, x], 1)

class transition_block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(transition_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        )
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        res = self.avg_pool(x)
        return res

class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = growth_rate * 2
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_layers(block, num_planes, num_blocks[0])
        num_planes += num_blocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = transition_block(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_layers(block, num_planes, num_blocks[1])
        num_planes += num_blocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = transition_block(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_layers(block, num_planes, num_blocks[2])
        num_planes += num_blocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = transition_block(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_layers(block, num_planes, num_blocks[3])
        num_planes += num_blocks[3]*growth_rate

        self.transform = nn.Sequential(
            nn.BatchNorm2d(num_planes),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4)
        )

        self.linear = nn.Linear(num_planes, num_classes)

    def _make_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            in_planes = self.growth_rate

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = self.transform(x)
        re = x.view(x.size(0), -1)
        res = self.linear(re)
        return res

def DenseNet121():
    return DenseNet(bottleneck_block, [6, 12, 24, 16], growth_rate=32)

def DenseNet169():
    return DenseNet(bottleneck_block, [6, 12, 32, 32], growth_rate=32)

def DenseNet201():
    return DenseNet(bottleneck_block, [6, 12, 48, 32], growth_rate=32)

def DenseNet161():
    return DenseNet(bottleneck_block, [6, 12, 36, 24], growth_rate=48)

def DenseNetCifar():
    return DenseNet(bottleneck_block, [6, 12, 24, 16], growth_rate=12)
