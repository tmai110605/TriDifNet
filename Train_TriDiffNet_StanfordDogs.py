#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriDiffNet training script for fine-grained Stanford Dogs and CIFAR benchmark datasets.
"""
import argparse
import sys
import os
import time
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets

from models.cross_entropy import LabelSmoothingCrossEntropy
from models.datasets import StanfordDogs
from models.TriDiffNet2 import build_TriDiffNet
import writeLogAcc as wA

def get_args():
    parser = argparse.ArgumentParser(
        description='TriDiffNet training script for StanfordDogs and fine-grained datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-r', '--data-root', type=str, default='./data', help='Dataset root path.')
    parser.add_argument('-d', '--dataset', type=str, choices=['cifar10', 'cifar100', 'dogs'], default='dogs', help='Dataset name.')
    parser.add_argument('-t', '--typesize', type=str, choices=['compact', 'ultra'], default='compact', help='Model typesize (compact: ~1.49M, ultra: ~1.05M).')
    parser.add_argument('--download', action='store_true', help='Download the specified dataset before running the training.')
    parser.add_argument('-g', '--gpu-id', default=0, type=int, help='ID of the GPU to use. Set to -1 to use CPU.')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size.')
    parser.add_argument('-e', '--epochs', default=200, type=int, help='Number of total epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, help='Initial learning rate.')
    parser.add_argument('-s', '--schedule', nargs='+', default=[100, 150, 180], type=int, help='Learning rate schedule epochs.')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='SGD weight decay.')
    parser.add_argument('--smoothing', default=0.1, type=float, help='Label smoothing factor (default: 0.1).')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on validation set.')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint for evaluation.')
    return parser.parse_args()


def get_device(args):
    if args.gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{args.gpu_id}')
    return torch.device('cpu')


def resolve_data_root(args):
    # Verify primary dataset directory
    if os.path.exists(args.data_root):
        return args.data_root
    # Fallback to shared dataset directory
    if os.path.exists('../../../datasets/StanfordDogs'):
        return '../../../datasets/StanfordDogs'
    return args.data_root


def get_data_loader(args, train):
    data_root = resolve_data_root(args)
    if args.dataset in ('cifar10', 'cifar100'):
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        dataset_class = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100
    elif args.dataset == 'dogs':
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])
        dataset_class = StanfordDogs
    else:
        raise NotImplementedError(f"Can't determine data loader for dataset '{args.dataset}'")

    # Instantiate dataset with automatic download support
    dataset = dataset_class(root=data_root, train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if train else False,
        num_workers=args.workers
    )


def calculate_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res


def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, args, device):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    batch_count = len(data_loader)
    losses = []
    accs1 = []
    accs5 = []
    for (n_batch, (images, target)) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        loss_item = loss.item()
        losses.append(loss_item)
        acc1, acc5 = calculate_accuracy(output, target, topk=(1, 5))
        accs1.append(acc1)
        accs5.append(acc5)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (n_batch % 10) == 0:
            mode_str = 'train' if train else ' val '
            print(f'[{mode_str}] epoch {n_epoch + 1}/{args.epochs}, batch {n_batch + 1}/{batch_count}, loss={loss_item:.5f}, top1={100.0 * acc1:.2f}%, top5={100.0 * acc5:.2f}%')

    return (sum(losses) / len(losses), sum(accs1) / len(accs1), sum(accs5) / len(accs5))


def main():
    args = get_args()
    print('Command: {}'.format(' '.join(sys.argv)))
    device = get_device(args)
    print(f'Using device: {device}')

    # torch.autograd.set_detect_anomaly(True)  # Disabled for maximum GPU training throughput

    dataset_name = 'StanfordDogs' if args.dataset == 'dogs' else args.dataset.upper()
    strmode = f'{dataset_name}_TriDiffNet_{args.typesize}'
    pathout = os.path.join('./checkpoints', strmode)
    filenameLOG = os.path.join(pathout, f'{strmode}.txt')
    if not os.path.exists(pathout):
        os.makedirs(pathout)

    num_classes = 120 if args.dataset == 'dogs' else (100 if args.dataset == 'cifar100' else 10)
    cifar = (args.dataset in ('cifar10', 'cifar100'))

    # Build model
    model = build_TriDiffNet(
        num_classes=num_classes,
        typesize=args.typesize,
        cifar=cifar,
        dropout_rate=0.2
    )
    model = model.to(device)
    print(model)

    params_count = sum(p.data.nelement() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable model parameters: {params_count:,} ({params_count / 1e6:.2f} M)')

    # Loss function, optimizer, scheduler
    if args.smoothing > 0:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
    else:
        train_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    val_criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=args.schedule,
        gamma=0.1
    )

    # Data loaders
    train_loader = get_data_loader(args=args, train=True)
    val_loader = get_data_loader(args=args, train=False)

    if args.evaluate:
        pathcheckpoint = args.checkpoint if args.checkpoint else os.path.join(pathout, 'model_best.pth')
        if os.path.isfile(pathcheckpoint):
            print(f"=> loading checkpoint '{pathcheckpoint}'")
            checkpoint = torch.load(pathcheckpoint, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            del checkpoint
        else:
            print(f"=> no checkpoint found at '{pathcheckpoint}'")
            return

        t_start = time.time()
        loss_val, acc_val, acc_val5 = run_epoch(
            train=False,
            data_loader=val_loader,
            model=model,
            criterion=val_criterion,
            optimizer=None,
            n_epoch=0,
            args=args,
            device=device
        )
        t_elapsed = time.time() - t_start
        eval_log = f'Evaluation Validation: Loss={loss_val:.5f}, Top1={100.0 * acc_val:.2f}%, Top5={100.0 * acc_val5:.2f}%, Time={t_elapsed:.2f}s'
        print(eval_log)
        wA.writeLogAcc(filenameLOG, eval_log)
        return

    # Training loop
    acc_val_max = 0.0
    for n_epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Starting epoch {n_epoch + 1}/{args.epochs}, lr={current_lr}')

        loss_train, acc_train, acc_train5 = run_epoch(
            train=True,
            data_loader=train_loader,
            model=model,
            criterion=train_loss_fn,
            optimizer=optimizer,
            n_epoch=n_epoch,
            args=args,
            device=device
        )

        loss_val, acc_val, acc_val5 = run_epoch(
            train=False,
            data_loader=val_loader,
            model=model,
            criterion=val_criterion,
            optimizer=None,
            n_epoch=n_epoch,
            args=args,
            device=device
        )

        log_str = f'Epoch {n_epoch+1:03d}: Train_Loss={loss_train:.4f}, Train_Top1={100.0*acc_train:.2f}% | Val_Loss={loss_val:.4f}, Val_Top1={100.0*acc_val:.2f}%, Val_Top5={100.0*acc_val5:.2f}%'
        print(log_str)
        wA.writeLogAcc(filenameLOG, log_str)

        if acc_val > acc_val_max:
            acc_val_max = acc_val
            best_model_path = os.path.join(pathout, 'model_best.pth')
            torch.save({
                'epoch': n_epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_acc': acc_val_max,
                'optimizer': optimizer.state_dict(),
            }, best_model_path)
            print(f"[*] New best model saved to {best_model_path} ({100.0 * acc_val_max:.2f}%)")

        scheduler.step()

if __name__ == '__main__':
    main()

