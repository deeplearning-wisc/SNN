import argparse
import os
import shutil
import time
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.densenet as dn



parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--id', default=None, type=str, required = True,
                    help='In dataset')
parser.add_argument('--temp', default=0.04, type=float,
                    help='Temperature')
parser.add_argument('--seed', default=0, type=int,
                    help='Random Seed')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', '--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet-101_cifar100', type=str,
                    help='name of experiment')
parser.add_argument('--r', default=None, type=float, help='relevance ratio', required = True)

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Data loading code
    #CIFAR10

    in_dataset = args.id

    if in_dataset == "CIFAR-10":
     normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    elif in_dataset == "CIFAR-100":
        normalize = transforms.Normalize(mean=[0.507,0.487,0.441],
                                      std=[0.267,0.256,0.276])
    else:
        raise Exception("Wrong Dataset")
    
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    

    kwargs = {'num_workers': 8, 'pin_memory': True}

    if in_dataset == "CIFAR-10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                            transform=transform_train),
            batch_size=args.bs, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.bs, shuffle=True, **kwargs)
        num_classes = 10
       

    elif in_dataset == "CIFAR-100":
        godin_name = "densenet100"
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                            transform=transform_train),
            batch_size=args.bs, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transform_test),
            batch_size=args.bs, shuffle=True, **kwargs)
        num_classes = 100
    else:
        raise Exception("Wrong Dataset")
   
    print(f"Loading {args.id} with num classes = {num_classes}")
    model = dn.DenseNet3(args.layers, num_classes=num_classes, growth_rate = 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer= None, r=args.r)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
   
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.epochs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename = f"checkpoint_{epoch}.pth.tar")
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
  #      print(loss, prec1)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
        #    print(loss, prec1)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        # log to TensorBoard
    
    return top1.avg


def save_checkpoint(args, state, is_best, filename):
    """Saves checkpoint to disk"""
    directory = os.path.join("./checkpoints", f"{args.id}","densenet")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory,filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory,'model_best.pth.tar'))

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


def adjust_learning_rate(optimizer, epoch, tot_epochs):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    if tot_epochs == 300:
         lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    elif tot_epochs == 100:
         lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 75)) * (0.1 ** (epoch // 90))
    else:
        raise Exception("Check Epochs")
    # log to TensorBoard
    print(f"Current lr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
