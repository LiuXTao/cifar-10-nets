import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
LeNet -> AlexNet -> VGG -> GoogleNet -> ResNet -> WideResNet -> ResNeXt -> DenseNet ->(SENet) ->DPN

缩聚-激发网络 

'''

class Basicblock(nn.Module):
    def __init__(self): 
        super(Basicblock, self).__init__()

