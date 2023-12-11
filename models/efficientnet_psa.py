import torch
from torch import nn
from torch.nn import functional as F

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        self.ca_sa = ChannelSpatialAttention(planes)

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

        out = self.ca_sa(out)
        return out

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * se_ratio)),
            Swish(),
            nn.Linear(int(in_channels * se_ratio), in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        out = self.squeeze(x).view(b, c)
        out = self.excitation(out).view(b, c, 1, 1)
        return x * out.expand_as(x)
    
class SEConvBlockPSA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, se_ratio, psa_planes, groups=1):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, groups)
        self.se = SEBlock(out_channels, se_ratio)
        self.psa = PSAModule(out_channels, psa_planes)  # 添加PSAModule

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.psa(x)  # 应用PSAModule
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.bn(self.conv(x)))

class SEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, se_ratio, groups=1):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, groups)
        self.se = SEBlock(out_channels, se_ratio)

    def forward(self, x):
        return self.se(self.conv(x))

class EfficientNet(nn.Module):
    def __init__(self, width_factor, depth_factor, scale, dropout_rate=0.2, num_classes=10):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        se_ratios = [0.25] * 7
        depth = depth_factor
        width = width_factor

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        self.stage1 = ConvBlock(3, channels[0], kernel_size=3, stride=2, padding=1)

        previous_channels = channels[0]
        self.stages = []
        for rep, chs, kernel_size, stride, se_ratio in zip(repeats, channels[1:], kernel_sizes, strides, se_ratios):
            for i in range(rep):
                if i == 0:
                    self.stages.append(SEConvBlockPSA(previous_channels, chs, kernel_size, stride, kernel_size//2, se_ratio, chs))
                else:
                    self.stages.append(SEConvBlockPSA(previous_channels, chs, kernel_size, 1, kernel_size//2, se_ratio, chs))
                previous_channels = chs
        self.stages = nn.Sequential(*self.stages)

        self.stage2 = ConvBlock(previous_channels, channels[-1], kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stages(out)
        out = self.stage2(out)
        out = self.avgpool(out).view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def EfficientB0_att(num_classes):
    # EfficientNet-B0
    model_b0 = EfficientNet(width_factor=1.0, depth_factor=1.0, scale=1.0, dropout_rate=0.2, num_classes=num_classes)

    return model_b0

def EfficientB1_att(num_classes):
    # EfficientNet-B1
    model_b1 = EfficientNet(width_factor=1.0, depth_factor=1.1, scale=1.2, dropout_rate=0.2, num_classes=num_classes)
    
    return model_b1

def EfficientB2_att(num_classes):
    # EfficientNet-B2
    model_b2 = EfficientNet(width_factor=1.1, depth_factor=1.2, scale=1.4, dropout_rate=0.3, num_classes=num_classes)
    return model_b2

def EfficientB3_att(num_classes):
    # EfficientNet-B3
    model_b3 = EfficientNet(width_factor=1.2, depth_factor=1.4, scale=1.8, dropout_rate=0.3, num_classes=num_classes)

    return model_b3

def EfficientB4_att(num_classes):
    # EfficientNet-B4
    model_b4 = EfficientNet(width_factor=1.4, depth_factor=1.8, scale=2.2, dropout_rate=0.4, num_classes=num_classes)

    return model_b4

