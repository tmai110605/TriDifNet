# -*- coding: utf-8 -*-
"""
TriDiffNet distributed / standard training and validation script on ImageNet-1k.
"""
import os
import argparse
import random
import shutil
import time
import warnings

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

from models.cross_entropy import LabelSmoothingCrossEntropy
from models.TriDiffNet2 import build_TriDiffNet
import writeLogAcc as wA

model_names = ['TriDiffNet_compact', 'TriDiffNet_ultra']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with TriDiffNet')
parser.add_argument('-r', '--data', type=str, default='../../../datasets/ImageNet', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='TriDiffNet_compact',
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--smoothing', default=0.1, type=float,
                    help='Label smoothing factor (default: 0.1).')
parser.add_argument('--use-diff', dest='use_diff', action='store_true', default=True,
                    help='Enable TensorDiff3D in every block (default: enabled).')
parser.add_argument('--no-diff', dest='use_diff', action='store_false',
                    help='Disable TensorDiff3D — pure MuS-FR baseline for ablation.')

best_acc1 = 0

def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size
        )

    # Determine typesize
    if 'ultra' in args.arch:
        typesize = 'ultra'
    else:
        typesize = 'compact'

    print(f"=> Creating model '{args.arch}' with typesize '{typesize}', use_diff={args.use_diff}")
    model = build_TriDiffNet(num_classes=1000, typesize=typesize, cifar=False, use_diff=args.use_diff)

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    elif torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    params_count = sum(p.data.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {params_count:,} ({params_count / 1e6:.2f} M)")

    if args.smoothing > 0:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        train_loss_fn = train_loss_fn.cuda(args.gpu)
        val_criterion = val_criterion.cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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
        validate(val_loader, model, val_criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, train_loss_fn, optimizer, epoch, args)
        acc1 = validate(val_loader, model, val_criterion, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'./checkpoints/{args.arch}_checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {loss.item():.4f}")


def validate(val_loader, model, criterion, args):
    model.eval()
    correct1 = 0
    total = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct1 += pred.eq(target.view(1, -1).expand_as(pred)).sum().item()
            total += target.size(0)

    acc1 = 100.0 * correct1 / total
    print(f" * Acc@1 {acc1:.3f}%")
    return acc1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
