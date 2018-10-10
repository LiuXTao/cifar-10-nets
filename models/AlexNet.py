import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
AlexNet: 5层卷积层+3层全连接层
 提特征层
 conv1: conv2d()-->ReLU()-->MaxPool2d()-->Batchnorm
 conv2: conv2d()-->ReLU()-->MaxPool2d()-->Batchnorm
 conv3: conv2d()-->ReLU()
 conv4: conv2d()-->ReLU()
 conv5: conv2d()-->ReLU()-->MaxPool2d()
 
 线性层
 full_connection_layer1: Linear()-->ReLU-->Dropout()
 full_connection_layer2: Linear()-->ReLU-->Dropout()
 full_connection_layer3: Linear()
 
 
 Alexnet的特点：
 1）使用ReLU作为CNN的激活函数，并验证其效果在较深的网络超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题;加快了训练速度（因为训练网络使用了梯度下降，非饱和的非线性函数训练速度快于饱和的非线性函数）。
 2）训练时，在全连接层后使用Dropout,随机忽略一些神经元，以避免模型过拟合
 3）CNN中使用重叠的最大池化，避免平均赤化的模糊化效果。并且AlexNet中指出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性
 4）提出了LRN层，对局部神经元的活动创建竞争机制，使得其中相应比较大的值变得相对更大，并一直其他反馈小的神经元，增强了模型的泛化能力
 5）数据增强。随机从256*256的原始图像中截取224*224大小的区域。进行数据增强可以大大减轻过拟合，提升泛化能力。因为仅靠原始数据量，参数众多的CNN会陷入过拟合中
  
'''

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )                #feature_map的大小计算公式（I-K+2P）/S+1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(192)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384,384,3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.full_layer6 = nn.Sequential(
            nn.Linear(256*2*2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.full_layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.full_layer8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        res = x.view(x.size(0), -1)
        res = self.full_layer6(res)
        res = self.full_layer7(res)
        res = self.full_layer8(res)

        return res

# if __name__ == '__main__':
#     model = AlexNet()
#     print(model)看到了