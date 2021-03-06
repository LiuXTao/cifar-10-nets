import torch.nn as nn
import torch

'''
LeNet-5有7层网络结构
第一层C1是一个卷积层 
输入图片: 32*32 
卷积核大小: 5*5 
卷积核种类: 6 
输出feature map大小:28*28(32-5+1) 
神经元数量:28*28*6 
可训练参数数量:(5*5+1)*6，(每个卷积核25个权重值w，一个截距值bias;总共6个卷积核) 
连接数量:(5*5+1)*6*28*28

 （ 包含S2是一个下采样层(池化层): 
输入:28*28 
采样区域:2*2 
采样方式:4个输入相加，乘以一个可训练参数，再加上一个可训练偏置，结果通过sigmoid。(论文原文是这样描述，但是实际中，我看到一般都是用最大池化) 
种类数量：6 
输出的feature map大小时:14*14(28/2) 
神经元数量:14*14*6 
可训练参数:2*6(和的权重w和偏置bias,然后乘以6) 
连接数:(2*2+1)*6*14*14
）

2.第二层C3也是一个卷积层 
输入:S2中所有6个或者几个特征的map组合，这个组合并无太大实际意义，受限于当时的硬件水平，才这样组合 
卷积核大小:5*5 
卷积核种类:16 
输出feature map大小:10*10 
C3中的每个特征map是连接到S2中的所有6个或者几个特征map的，表示本层的特征map是上一层提取到的 
特征map的不同组合,存在的一个方式是：C3的前6个特征图以S2中3个相邻的特征图子集为输入。接下来 
6个特征图以S2中4个相邻特征图子集为输入。然后的3个以不相邻的4个特征图子集为输入。最后一个 
将S2中所有特征图为输入。此时可训练参数:6*(3*25+1)+6*(4*25+1)+3*(4*25+1)+(25*6+1)=1516 
连接数：10*10*1516=151600

 （包含S4是一个下采样层(池化层) 
输入:10*10 
采样区域:2*2 
采样方式:4个输入相加，乘以一个可训练参数，再加上一个可训练偏置,结果通过sigmoid 
采样种类:16 
输出feature map大小:5*5(10/2) 
神经元数量:5*5*16=400 
可训练参数:2*16=32(和的权重2+偏置bias,乘以16) 
连接数:16*(2*2+1)*5*5=2000
）
3.第三层C5是一个卷积层(论文原文的描述) 
输入:S4层的全部16个单元特征map(与S4全连接) 
卷积核大小:5*5 
卷积核种类:120 
输出feature map大小:1*1 
可训练参数/连接数:120*(16*5*5+1)=48120

4.第四层F6层全连接层 
输入:C5 120维向量 
计算方式:计算输入向量和权重向量之间的点积，再加上一个偏置，结果通过sigmoid函数 
可训练参数:84*(120+1)=10164

5.第五层是F7全连接层

'''


class LeNet(nn.Module):
    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(       # heature_map size:32*32
            nn.Conv2d(3, 6, 5),           #  feature_map :28*28: 6
            nn.AvgPool2d(2,2),             #  14*14:3
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),          # 10*10 :16
            nn.AvgPool2d(2,2),              #5*5:16
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(16*5*5, 120)    # 此处5*5是feature_map的大小, 16表示有16个特征图
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        re = x.view(x.size(0), -1)
        x = self.fc1(re)
        x = self.fc2(x)
        res = self.fc3(x)
        return res
