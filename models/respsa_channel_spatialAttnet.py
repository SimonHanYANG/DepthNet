import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .SE_weight_module import SEWeightModule

# import cnsn
from .cnsn import CrossNorm, SelfNorm, CNSN 

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelSpatialAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 16, in_planes, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        return ca * sa * x


class PSAModule(nn.Module):
    def __init__(self, inplanes, planes, conv_kernels=[3, 5, 7], stride=1, conv_groups=[1, 8, 16]):
        super(PSAModule, self).__init__()

        assert len(conv_kernels) == len(conv_groups), "conv_kernels and conv_groups should have same length."

        self.conv_branches = nn.ModuleList()
        for kernel_size, groups in zip(conv_kernels, conv_groups):
            padding = kernel_size // 2
            self.conv_branches.append(
                nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
            )

        self.bn = nn.BatchNorm2d(planes * len(conv_kernels))
        self.relu = nn.ReLU(inplace=True)
        self.ca_sa = ChannelSpatialAttention(planes * len(conv_kernels))

    def forward(self, x):
        features = []
        for conv in self.conv_branches:
            features.append(conv(x))

        out = torch.cat(features, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.ca_sa(out)
        return out
    
class ResidualpsattBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_kernels=[3, 5, 7]):
        super(ResidualpsattBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.psa = PSAModule(out_channels, out_channels, conv_kernels=conv_kernels)
        self.conv2 = nn.Conv2d(out_channels * len(conv_kernels), out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.psa(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RespsaCSAttNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(RespsaCSAttNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def respsacsattnet18(num_classes):
    return RespsaCSAttNet(ResidualpsattBlock, [2,2,2,2], num_classes=num_classes)

def respsacsattnet50(num_classes):
    return RespsaCSAttNet(ResidualpsattBlock, [3,4,6,3], num_classes=num_classes)

def respsacsattnet101(num_classes):
    return RespsaCSAttNet(ResidualpsattBlock, [3,4,23,3], num_classes=num_classes)

'''
P.s. 没有 SE Weight 模块
'''

'''
在这个示例中，ChannelSpatialAttention 模块首先通过 channel_attention 子模块
计算通道注意力，然后通过 spatial_attention 子模块计算空间注意力。
最后，它将两种注意力进行元素级别的乘法，然后再将结果乘以输入特征图 x，
从而同时考虑了通道和空间的注意力。
在 PSAModule 的 forward 函数中，我们在所有卷积分支的输出合并和激活函数之后，
添加了 ChannelSpatialAttention 模块。
'''

'''
ChannelSpatialAttention 示意图：

输入是尺寸为 (batch_size, C, H, W) 的特征图。

首先，这个特征图会经过一个通道注意力（Channel Attention）模块，这个模块会计算每个通道的权重。这个过程可以被表示为一个从输入特征图向下的箭头，然后是一个全局平均池化（Global Average Pooling）操作，接着是两个 1x1 卷积（Convolution）操作，最后是一个 sigmoid 激活函数。输出是一个尺寸为 (batch_size, C, 1, 1) 的通道权重图。

同时，输入特征图也会经过一个空间注意力（Spatial Attention）模块，这个模块会计算每个空间位置的权重。这个过程可以被表示为一个从输入特征图向右的箭头，然后是一个通道平均池化（Channel Average Pooling）和通道最大池化（Channel Max Pooling）操作，接着是一个 3x3 卷积操作，最后是一个 sigmoid 激活函数。输出是一个尺寸为 (batch_size, 1, H, W) 的空间权重图。

最后，通道权重图和空间权重图会被同时应用到输入特征图上。这个过程可以被表示为一个从通道权重图和空间权重图向上的箭头，然后是一个元素相乘（Element-wise Multiplication）操作，最后输出尺寸为 (batch_size, C, H, W) 的注意力应用后的特征图。
'''
'''

通过 ChannelSpatialAttention 构建的 ResidualBlock 示意图：

输入是尺寸为 (batch_size, C, H, W) 的特征图。

首先，这个特征图会经过一个卷积（Convolution）和批次标准化（Batch Normalization）操作，然后会经过一个激活函数（Activation Function）。这个过程可以被表示为一个从输入特征图向下的箭头，然后是一个卷积操作，接着是一个批次标准化操作，最后是一个激活函数。

然后，这个特征图会经过一个 ChannelSpatialAttention 模块，这个过程可以被表示为一个从上一步的输出向下的箭头，然后是一个 ChannelSpatialAttention 模块。

接着，注意力应用后的特征图会经过另一个卷积和批次标准化操作。这个过程可以被表示为一个从 ChannelSpatialAttention 模块的输出向下的箭头，然后是一个卷积操作，接着是一个批次标准化操作。

同时，如果输入特征图的通道数或尺寸需要改变，那么输入特征图也会经过一个卷积和批次标准化操作。这个过程可以被表示为一个从输入特征图向右的箭头，然后是一个卷积操作，接着是一个批次标准化操作。

最后，上述两个步骤的输出会被相加，然后经过一个激活函数。这个过程可以被表示为一个从上述两个步骤向上的箭头，然后是一个加法操作，最后是一个激活函数。

'''