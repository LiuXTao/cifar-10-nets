import torch
import torch.nn as nn
import math
'''
残差神经网络：


'''

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)
        res = self.relu2(x+residual)
        return res

# 瓶颈结构
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes*4)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        res = self.relu(x + residual)
        return res

class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512*block.expansion, num_class)

        # 模块初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] *m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        re = x.view(x.size(0), -1)
        res = self.fc(re)
        return res


    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
         # 不同 channel时， 进行维度变换
        if stride!=1 or self.in_planes != planes*block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, down_sample))
        self.in_planes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(BasicBlock, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)



