import torch
from torch import nn, optim

class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.down_sample(x)
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.dense_layers = nn.Sequential(
            *[BottleNeck(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)]
        )
        
    def forward(self, x):
        return self.dense_layers(x)
    
class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate=32, num_classes=1000, theta=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        channels = 2 * growth_rate
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for num_layers in block_config:
            block = DenseBlock(channels, growth_rate, num_layers)
            self.dense_blocks.append(block)
            channels += growth_rate * num_layers
            out_channels = int(channels * theta)
            self.transitions.append(Transition(channels, out_channels))
            channels = out_channels
        self.final_bn = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        for block, transition in zip(self.dense_blocks, self.transitions):
            x = transition(block(x))
        x = self.final_bn(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def Densenet121(num_classes):
    return DenseNet([6, 12, 24, 16], num_classes=num_classes)

def Densenet161(num_classes):
    return DenseNet([6, 12, 36, 24], growth_rate=48, num_classes=num_classes)

def Densenet169(num_classes):
    return DenseNet([6, 12, 32, 32], num_classes=num_classes)

def Densenet201(num_classes):
    return DenseNet([6, 12, 48, 32], num_classes=num_classes)