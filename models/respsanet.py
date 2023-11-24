import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .SE_weight_module import SEWeightModule

# import cnsn
from .cnsn import CrossNorm, SelfNorm, CNSN 

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

    
class ResidualpsaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 16]):
        super(ResidualpsaBlock, self).__init__()

        self.PSAModule1 = PSAModule(in_channels, out_channels, conv_kernels, stride, conv_groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.PSAModule2 = PSAModule(out_channels, out_channels, conv_kernels, 1, conv_groups)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.PSAModule1(x)))
        out = self.bn2(self.PSAModule2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RespsaNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(RespsaNet, self).__init__()
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

def respsanet18(num_classes):
    return RespsaNet(ResidualpsaBlock, [2,2,2,2], num_classes=num_classes)

def respsanet50(num_classes):
    return RespsaNet(ResidualpsaBlock, [3,4,6,3], num_classes=num_classes)

def respsanet101(num_classes):
    return RespsaNet(ResidualpsaBlock, [3,4,23,3], num_classes=num_classes)