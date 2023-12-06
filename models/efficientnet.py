import torch
from torch import nn
from torch.nn import functional as F

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
                    self.stages.append(SEConvBlock(previous_channels, chs, kernel_size, stride, kernel_size//2, se_ratio))
                else:
                    self.stages.append(SEConvBlock(previous_channels, chs, kernel_size, 1, kernel_size//2, se_ratio))
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

def EfficientB0(num_classes):
    # EfficientNet-B0
    model_b0 = EfficientNet(width_factor=1.0, depth_factor=1.0, scale=1.0, dropout_rate=0.2, num_classes=num_classes)

    return model_b0

def EfficientB1(num_classes):
    # EfficientNet-B1
    model_b1 = EfficientNet(width_factor=1.0, depth_factor=1.1, scale=1.2, dropout_rate=0.2, num_classes=num_classes)
    
    return model_b1

def EfficientB2(num_classes):
    # EfficientNet-B2
    model_b2 = EfficientNet(width_factor=1.1, depth_factor=1.2, scale=1.4, dropout_rate=0.3, num_classes=num_classes)
    return model_b2

def EfficientB3(num_classes):
    # EfficientNet-B3
    model_b3 = EfficientNet(width_factor=1.2, depth_factor=1.4, scale=1.8, dropout_rate=0.3, num_classes=num_classes)

    return model_b3

def EfficientB4(num_classes):
    # EfficientNet-B4
    model_b4 = EfficientNet(width_factor=1.4, depth_factor=1.8, scale=2.2, dropout_rate=0.4, num_classes=num_classes)

    return model_b4

