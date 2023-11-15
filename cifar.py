import os
import torch
from torchvision import datasets
from torchvision.transforms import transforms

def CIFAR10Dataset(traindir, valdir, transform):
    
    # Define a transform to normalize the data
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # cifar_dataroot = "./cifar_data/"

    if not os.path.exists(traindir):
        os.makedirs(traindir)
        os.makedirs(valdir)

        # Download and load the training data
        trainset = datasets.CIFAR10(root=traindir, train=True, download=True, transform=transform)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

        # Download and load the testing data
        testset = datasets.CIFAR10(root=valdir, train=False, download=True, transform=transform)
        # testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

    else:
        trainset = datasets.CIFAR10(root=traindir, train=True, download=False, transform=transform)
        testset = datasets.CIFAR10(root=valdir, train=False, download=False, transform=transform)
        
    
    return trainset, testset
