import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from .alexnet import *

from .vgg import *

from .mobilenetv2 import *

from .resnet import *

from .googlenet import *

from .shufflenet import *

from .efficientnet import *

from .densenet import *

from .shufflenet_psa import *

from .efficientnet_psa import *

# too much conv for cifia10
def alexnet(num_classes):
    # Load the not pretrained model
    # model = models.alexnet(weights="MAGENET1K_V1")
    # Change the last layer to match the number of classes in your task
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # create own alexnet
    model = AlexNet(num_classes)
    
    return model

def vgg16(num_classes):
    # Load the pretrained model
    # model_vgg16 = models.vgg16(weights="IMAGENET1K_V1")
    # Change the last layer to match the number of classes in your task
    # model_vgg16.classifier[6] = nn.Linear(model_vgg16.classifier[6].in_features, num_classes)
    
    # create own vgg16
    model_vgg16 = VGG16(num_classes)
    
    return model_vgg16
    

def vgg19(num_classes):
    # model_vgg19 = models.vgg19(weights="MAGENET1K_V1")
    # Change the last layer to match the number of classes in your task
    # model_vgg19.classifier[6] = nn.Linear(model_vgg19.classifier[6].in_features, num_classes)

    model_vgg19 = VGG19(num_classes)
    
    return model_vgg19
    
def resnet18(num_classes):
    # model_resnet18 = models.resnet18(weights="MAGENET1K_V1")
    # model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, num_classes)

    model_resnet18 = ResNet18(num_classes=num_classes)
    
    return model_resnet18
    
def resnet34(num_classes):
    # model_resnet34 = models.resnet34(weights="MAGENET1K_V1")
    # model_resnet34.fc = nn.Linear(model_resnet34.fc.in_features, num_classes)

    model_resnet34 = ResNet34(num_classes=num_classes)

    return model_resnet34
    
def resnet50(num_classes):
    # model_resnet50 = models.resnet50(weights="MAGENET1K_V1")
    # model_resnet50.fc = nn.Linear(model_resnet50.fc.in_features, num_classes)

    model_resnet50 = ResNet50(num_classes=num_classes)

    return model_resnet50

def resnet101(num_classes):
    # model_resnet101 = models.resnet101(weights="MAGENET1K_V1")
    # model_resnet101.fc = nn.Linear(model_resnet101.fc.in_features, num_classes)

    model_resnet101 = ResNet101(num_classes=num_classes)

    return model_resnet101

def mobilenetv2(num_classes):
    # model = models.mobilenet_v2(weights="MAGENET1K_V1")
    # Change the last layer to match the number of classes in your task
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = MobileNetV2(num_classes)
    
    return model

def googlenet(num_classes):
    model = GoogleNet(num_classes=num_classes)

    return model

def shufflenet_v2_x0_5(num_classes):
    model = Shufflenet_v2_x0_5(num_classes=num_classes)

    return model

def shufflenet_v2_x1_0(num_classes):
    model = Shufflenet_v2_x1_0(num_classes=num_classes)

    return model

def shufflenet_v2_x1_5(num_classes):
    model = Shufflenet_v2_x1_5(num_classes=num_classes)

    return model

def shufflenet_v2_x2_0(num_classes):
    model = Shufflenet_v2_x2_0(num_classes=num_classes)

    return model

def shufflenet_v2_x0_5(num_classes):
    model = Shufflenet_v2_x0_5_att(num_classes=num_classes)

    return model

def shufflenet_v2_x1_0(num_classes):
    model = Shufflenet_v2_x1_0_att(num_classes=num_classes)

    return model

def shufflenet_v2_x1_5(num_classes):
    model = Shufflenet_v2_x1_5_att(num_classes=num_classes)

    return model

def shufflenet_v2_x2_0(num_classes):
    model = Shufflenet_v2_x2_0_att(num_classes=num_classes)

    return model

def efficientb0(num_classes):
    # EfficientNet-B0
    model_b0 = EfficientB0(num_classes=num_classes)

    return model_b0

def efficientb1(num_classes):
    # EfficientNet-B1
    model_b1 = EfficientB1(num_classes=num_classes)
    
    return model_b1

def efficientb2(num_classes):
    # EfficientNet-B2
    model_b2 = EfficientB2(num_classes=num_classes)
    return model_b2

def efficientb3(num_classes):
    # EfficientNet-B3
    model_b3 = EfficientB3(num_classes=num_classes)

    return model_b3

def efficientb4(num_classes):
    # EfficientNet-B4
    model_b4 = EfficientB4(num_classes=num_classes)

    return model_b4

def efficientb0(num_classes):
    # EfficientNet-B0
    model_b0 = EfficientB0_att(num_classes=num_classes)

    return model_b0

def efficientb1(num_classes):
    # EfficientNet-B1
    model_b1 = EfficientB1_att(num_classes=num_classes)
    
    return model_b1

def efficientb2(num_classes):
    # EfficientNet-B2
    model_b2 = EfficientB2_att(num_classes=num_classes)
    return model_b2

def efficientb3(num_classes):
    # EfficientNet-B3
    model_b3 = EfficientB3_att(num_classes=num_classes)

    return model_b3

def efficientb4(num_classes):
    # EfficientNet-B4
    model_b4 = EfficientB4_att(num_classes=num_classes)

    return model_b4

# densenet too deep for cifia10
def densenet121(num_classes):
    model = Densenet121(num_classes=num_classes)

    return model

def densenet161(num_classes):
    model = Densenet161(num_classes=num_classes)

    return model


def densenet169(num_classes):
    model = Densenet169(num_classes=num_classes)

    return model

def densenet201(num_classes):
    model = Densenet201(num_classes=num_classes)

    return model

