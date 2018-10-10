import torch.nn as nn
import torch
'''
VGG-net 
通过反复堆叠3*3的小型卷积核和2*2的maxpool层

https://ss2.baidu.com/6ONYsjip0QIZ8tyhnq/it/u=2450254934,3041339605&fm=173&app=25&f=JPEG?w=600&h=604&s=CE8A742B51BFE1EB487D51CF0100A0B1

虽然从A到E每一级网络逐渐变深，但是网络的参数量并没有增长很多，这是因为参数量主要都消耗在最后3个全连接层。
前面的卷积部分虽然很深，但是消耗的参数量不大，不过训练比较耗时的部分依然是卷积，因其计算量比较大。

对于较浅的网络，如网络A，可以直接使用随机数进行随机初始化，而对于比较深的网络，则使用前面已经训练好的较浅的网络中的参数值对其前几层的卷积层和最后的全连接层进行初始化。

VGGNet在训练时有一个小技巧，先训练级别A的简单网络，再复用A网络的权重来初始化后面的几个复杂模型，这样训练收敛的速度更快

在训练的过程中，VGG比AlexNet收敛的要快一些， 
原因为：（1）使用小卷积核和更深的网络进行的正则化, 起到隐式规则话作用；（2）在特定的层使用了预训练（pre-train）得到的数据进行参数的初始化(initialize)。

VGG与Alexnet相比，具有如下改进几点：

1.去掉了LRN层，作者发现深度网络中LRN的作用并不明显，干脆取消了
2.采用更小的卷积核-3x3，Alexnet中使用了更大的卷积核，比如有7x7的，因此VGG相对于Alexnet而言，参数量更少.
(使用大卷积核会带来参数量的爆炸不说，而且图像中会存在一些部分被多次卷积，可能会给特征提取带来困难，所以在VGG中，普遍使用3x3的卷积。
 最后三层全连接层占用了很大一部分数据量，为了减少参数量，后几层的全连接网络都被全剧平均池化（global average pooling）和卷积操作代替了，但是全局平均池化也有很大的优点

)

3.池化核变小，VGG中的池化核是2x2，stride为2，Alexnet池化核是3x3，步长为2


---------------------


'''

# network structure config
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_class, batch_norm=True):
        super(VGG, self).__init__()
        self.features_layer = self._make_layers(cfg[vgg_name])
        self.classifier_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*1*1, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )
        self.batch_norm = batch_norm

    def forward(self, x):
        x = self.features_layer(x)
        # 展平
        re = x.view(x.size(0), -1)
        res = self.classifier_layer(re)
        return res

    def _make_layers(self, vgg_type):
        layers = []
        in_channel = 3
        for x in vgg_type:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif self.batch_norm == True:
                layers += [nn.Conv2d(in_channels=in_channel, out_channels=x, kernel_size=3,padding=1, stride=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
            else:
                layers += [nn.Conv2d(in_channels=in_channel, out_channels=x, kernel_size=3, padding=1, stride=1),
                           nn.ReLU()]
            in_channel = x
        return nn.Sequential(*layers)

def VGG11_btn(num_class):
    return VGG('VGG11', num_class=num_class, batch_norm=True)

def VGG11(num_class):
    return VGG('VGG11', num_class=num_class, batch_norm=False)

def VGG13_btn(num_class):
    return VGG('VGG13', num_class=num_class, batch_norm=True)

def VGG13(num_class):
    return VGG('VGG13', num_class=num_class, batch_norm=False)

def VGG16_btn(num_class):
    return VGG('VGG16', num_class=num_class, batch_norm=True)

def VGG16(num_class):
    return VGG('VGG16', num_class=num_class, batch_norm=False)

def VGG19_btn(num_class):
    return VGG('VGG119', num_class=num_class, batch_norm=True)

def VGG19(num_class):
    return VGG('VGG19', num_class=num_class, batch_norm=False)
