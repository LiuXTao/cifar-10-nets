import torch.nn as nn
import torch

'''
Google Net网络
=====>获得高质量模型最保险的做法就是增加模型的深度（层数）或者是其宽度（层核或者神经元数
甚至深层卷积神经网络会出现的缺陷·
1.参数太多，若训练数据集有限，容易过拟合；

2.网络越大计算复杂度越大，难以应用；

3.网络越深，梯度越往后穿越容易消失，难以优化模型。   

goal: 既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能

提出了BN方法：使大型卷积网络的训练速度加快，同时收敛后的分类准确率也得到提高
 会对每一个最小批数据的内部进行标准化处理，使输出规范化到N(0,1)正态分布，减少内部神经元分布的改变(internal Convatiate Shift)。
                        ===》 传统深度神经网络在训练时，每一个层输入的分布都在变化，导致训练变得困难。


'''
# 模块定义
class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU()
        )
        #
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(),
            nn.Conv2d(in_channels=kernel_3_in, out_channels=kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU()
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class GoogleNet(nn.Module):
    def __init__(self, num_class):
        super(GoogleNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128,192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        res = self.linear(x)
        return res

def google_net():
    return GoogleNet(10)





