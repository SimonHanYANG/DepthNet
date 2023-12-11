import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from loss import CrossEntropyLabelSmooth
from torchvision.utils import make_grid
import torch.nn.functional as F

from models.hannet import *

from torch.utils.tensorboard import SummaryWriter

import models
from cifar import CIFAR10Dataset
from cifar100 import CIFAR100Dataset
from CellDataset import CellDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# choosenable: ['conv', 'conv1x1', 'epsanet101', 'epsanet101_cn', 'epsanet50', 'epsanet50_cn']
# choosenable: ['conv', 'conv1x1', 'epsanet101', 'epsanet101_cn', 'epsanet50', 'epsanet50_cn', 'respsacsattnet101', 'respsacsattnet18', 'respsacsattnet50']
# ['alexnet', 'conv', 'conv1x1', 'epsanet101', 'epsanet101_cn', 'epsanet50', 
# 'epsanet50_cn', 'hannet101', 'hannet18', 'hannet34', 'hannet50', 
# 'make_layer', 'mobilenetv2', 'resnet101', 'resnet18', 'resnet34', 
# 'resnet50', 'respsacsattnet101', 'respsacsattnet18', 'respsacsattnet50', 
# 'vgg16', 'vgg19']
parser.add_argument('--arch', '-a', metavar='ARCH', default='epsanet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: epsanet50)')
parser.add_argument('--data', metavar='DIR',default='./dataset',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10', 
                    help="categroy of dataset")
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--nc', '--num-classes', default=100, type=int,
                    metavar='NC', help='number of classes (default: 100 for cifar100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int, nargs='+',
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--action', default='', type=str,
                    help='other information.')
parser.add_argument('--saf', '--saveattentionfig', default=False, type=bool,
                    help='path to latest checkpoint (default: none)')

best_prec1 = 0
best_prec5 = 0
best_epoch = 0


import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    num_class = args.nc
    print("Number of classes: " + str(num_class))
    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_class)
    else:
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](num_class)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print(model)

    # get the number of models parameters
    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    # criterion = CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1)
    criterion = CrossEntropyLabelSmooth(num_classes=num_class, epsilon=0.1)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # lr adapt change
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == "hs73":
        # using --data ./hs73_data/ --dataset hs73
        
        transform = transforms.Compose([
            # image size: (96, 96)
            transforms.Resize((96, 96)),  # 将所有图像调整为同一大小 (96, 96)
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            # transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
            # transforms.RandomRotation(10),  # 在 (-10, 10) 范围内随机旋转图像
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变图像的亮度、对比度、饱和度和色相
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # './cellData/train'
        traindir = os.path.join(args.data, 'train')
        # './cellData/val'
        valdir = os.path.join(args.data, 'eval')

        # create train/val dataset
        train_dataset = CellDataset(root_dir=traindir, transform=transform)
        val_dataset = CellDataset(root_dir=valdir, transform=transform)
        
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    elif args.dataset == "hs101":
        # using --data ./hs_data/ --dataset hs101
        
        transform = transforms.Compose([
            # image size: (96, 96)
            transforms.Resize((96, 96)),  # 将所有图像调整为同一大小 (96, 96)
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            # transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
            # transforms.RandomRotation(10),  # 在 (-10, 10) 范围内随机旋转图像
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变图像的亮度、对比度、饱和度和色相
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # './cellData/train'
        traindir = os.path.join(args.data, 'train')
        # './cellData/val'
        valdir = os.path.join(args.data, 'eval')

        # create train/val dataset
        train_dataset = CellDataset(root_dir=traindir, transform=transform)
        val_dataset = CellDataset(root_dir=valdir, transform=transform)
        
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    elif args.dataset == "hs91":
        # using --data ./human_sperm_data/ --dataset hs91
        
        transform = transforms.Compose([
            # image size: (96, 96)
            transforms.Resize((96, 96)),  # 将所有图像调整为同一大小 (96, 96)
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            # transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
            # transforms.RandomRotation(10),  # 在 (-10, 10) 范围内随机旋转图像
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变图像的亮度、对比度、饱和度和色相
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # './cellData/train'
        traindir = os.path.join(args.data, 'train')
        # './cellData/val'
        valdir = os.path.join(args.data, 'eval')

        # create train/val dataset
        train_dataset = CellDataset(root_dir=traindir, transform=transform)
        val_dataset = CellDataset(root_dir=valdir, transform=transform)
        
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    elif args.dataset == "sperm":
        # using --data ./cellData/ --dataset sperm
        
        transform = transforms.Compose([
            # image size: (96, 96)
            transforms.Resize((96, 96)),  # 将所有图像调整为同一大小 (96, 96)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # './cellData/train'
        traindir = os.path.join(args.data, 'train')
        # './cellData/val'
        valdir = os.path.join(args.data, 'val')

        # create train/val dataset
        train_dataset = CellDataset(root_dir=traindir, transform=transform)
        val_dataset = CellDataset(root_dir=valdir, transform=transform)
        
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
        
        
    elif args.dataset == "cifar10":
        # using --data ./dataset_cifar10/

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        
        train_dataset, val_datset = CIFAR10Dataset(traindir, valdir, transform)
            
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_datset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
        
    elif args.dataset == "cifar100":
        # using --data ./dataset_cifar100/

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        
        train_dataset, val_datset = CIFAR100Dataset(traindir, valdir, transform)
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_datset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        
        if os.path.exists(traindir):
            os.makedirs(traindir)
        
        if os.path.exists(valdir):
            os.makedirs(valdir)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        m = time.time()
        _, _ = validate(val_loader, model, criterion)
        n = time.time()
        print((n - m) / 3600)
        return

    directory = "runs/%s/" % (args.arch + '_' + args.action)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # create a SummaryWriter object
    writer_dir = directory + "/log/"
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    writer = SummaryWriter(writer_dir)
    
    # create attention map save folder
    att_map_folder = directory + "/attmap/"
    if not os.path.exists(att_map_folder):
        os.makedirs(att_map_folder)

    Loss_plot = {}
    train_prec1_plot = {}
    train_prec5_plot = {}
    val_prec1_plot = {}
    val_prec5_plot = {}

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch)
        loss_temp, train_prec1_temp, train_prec5_temp = train(train_loader, model, criterion, optimizer, epoch, writer, att_map_folder)
        Loss_plot[epoch] = loss_temp
        train_prec1_plot[epoch] = train_prec1_temp
        train_prec5_plot[epoch] = train_prec5_temp

        # evaluate on validation set
        # prec1 = validate(val_loader, model, criterion)
        prec1, prec5 = validate(val_loader, model, criterion, epoch, writer)
        val_prec1_plot[epoch] = prec1
        val_prec5_plot[epoch] = prec5

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        if is_best:
            best_epoch = epoch + 1
            best_prec5 = prec5
        print(' * BestPrec so far@1 {top1:.3f} @5 {top5:.3f} in epoch {best_epoch}'.format(top1=best_prec1,
                                                                                           top5=best_prec5,
                                                                                           best_epoch=best_epoch))

        data_save(directory + 'Loss_plot.txt', Loss_plot)
        data_save(directory + 'train_prec1.txt', train_prec1_plot)
        data_save(directory + 'train_prec5.txt', train_prec5_plot)
        data_save(directory + 'val_prec1.txt', val_prec1_plot)
        data_save(directory + 'val_prec5.txt', val_prec5_plot)

        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        print("-" * 80)
        print(time_value)
        print("-" * 80)
        writer.close()
        # scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, writer, att_map_folder):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_batch = {}
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # input: torch.Size([bs, 3, 32, 32])
        # target: torch.Size([bs])
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        
        loss = criterion(output, target)

        # write loss in tensorboard
        writer.add_scalar('Training loss', loss.item(), epoch)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            
        # save training epoech attention map
        # Randomly select 10 images to visualize attention maps for hannet
        if "hannet" in args.arch:
            # 如果您想在每个epoch结束时打印特征图，请在此处添加代码
            if i == len(train_loader) - 1:  # 检查是否为最后一个batch
                # 假设我们想要可视化PSAModule的输出
                # 将`visualize_feature_maps`设置为True以提取和可视化特征图
                visualize_feature_maps = True  
                if visualize_feature_maps:
                    # 确保模型是在评估模式
                    model.eval()
                    
                    # 正向传播来获取特征图
                    with torch.no_grad():
                        # 此处假设PSAModule是ResBlock的一部分并且我们使用了forward hooks
                        activation = {}
                        def get_activation(name):
                            def hook(model, input, output):
                                activation[name] = output.detach()
                            return hook
                        
                        # 注册hook
                        for name, m in model.named_modules():
                            if isinstance(m, PSAModule):
                                m.register_forward_hook(get_activation(name))
                        
                        # 正向传播获取特征图
                        _ = model(input)

                        # 为每个PSAModule可视化特征图
                        for name, act in activation.items():
                            # 选择一个特征图进行可视化
                            feature_map = act[0]  # 假设我们只可视化batch中的第一个特征图
                            
                            # 选择特征图的通道
                            for idx in range(feature_map.size(0)):
                                # 可视化特征图
                                plt.figure(figsize=(20, 10))
                                plt.imshow(feature_map[idx].cpu().numpy(), cmap='hot')
                                plt.colorbar()
                                plt.title(f"{name} Feature Map at Epoch {epoch}, Channel {idx}")
                                plt.savefig(f"{att_map_folder}/_{name}_FeatureMap_Epoch{epoch}_Channel{idx}.png")
                                plt.close()  # 关闭图形，以免消耗过多资源
                    model.train()  # 恢复训练模式

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
           
            # write loss in tensorboard
            writer.add_scalar('Testing loss', loss.item(), epoch)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.arch + '_' + args.action)

    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            '''
            Error:
            RuntimeError: view size is not compatible with input tensor's size and stride 
            (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            '''
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def data_save(root, file):
    if not os.path.exists(root):
        os.mknod(root)
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


if __name__ == '__main__':
    main()

