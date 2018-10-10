
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Dual Path Networks（DPN）

'''

class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer, cardinality):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Sequential(
            nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes+dense_depth)
        )

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        residual = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([residual[:, :d,:,:]+out[:,:d,:,:], residual[:, d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out

class DPN(nn.Module):
    def __init__(self, cfg, cardinality, num_classes=10):
        super(DPN, self).__init__()
        self.in_planes, self.out_planes, self.num_blocks, self.dense_depth = cfg['in_plane'], cfg['out_plane'], cfg['num_blocks'], cfg['dense_depth']

        self.cardinality = cardinality
        self.planes = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(Bottleneck, self.in_planes[0], self.out_planes[0], self.num_blocks[0], self.dense_depth[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, self.in_planes[1], self.out_planes[1], self.num_blocks[1], self.dense_depth[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, self.in_planes[2], self.out_planes[2], self.num_blocks[2], self.dense_depth[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, self.in_planes[3], self.out_planes[3], self.num_blocks[3], self.dense_depth[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(self.out_planes[3]+(self.num_blocks[3]+1)*self.dense_depth[3], num_classes)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.planes, in_planes, out_planes, dense_depth, stride, i==0, cardinality=self.cardinality))
            self.planes = out_planes + (i+2)*dense_depth
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DPN26():
    cfg = {
        'in_plane':(96, 192, 384, 768),
        'out_plane':(256, 512, 1024, 2048),
        'num_blocks':(2, 2, 2, 2),
        'dense_depth':(16, 32, 24, 128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_plane':(96, 192, 384, 768),
        'out_plane':(256, 512, 1024, 2048),
        'num_blocks':(3, 4, 20, 3),
        'dense_depth':(16, 32, 24, 128)
    }
    return DPN(cfg)


