from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from cifar10_models.config import args_wideresnet, args_preactresnet18,args_vgg16,args_resnet50,args_resnext,args_vgg19,args_senet,args_densenet121
from cifar10_models.utils import load_model, AverageMeter, accuracy
import torchvision

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('data.npy')
        labels = np.load('label.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    # loss = torch.nn.KLDivLoss(reduction='batchmean')
    # return loss(F.log_softmax(outputs, dim=1), smooth_labels)
    loss = nn.CrossEntropyLoss()
    return loss(outputs, smooth_labels)


def load_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='cifar10_models/data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=args['num_workers'])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='cifar10_models/data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                             shuffle=False, num_workers=args['num_workers'])
    return trainloader, testloader

def main():
    for arch in ['preactresnet18', 'wideresnet',"resnet50","densenet121","vgg16","vgg19"]:
        if arch == 'wideresnet':
            args = args_wideresnet
        elif arch == "resnet50":
            args = args_resnet50
        elif arch == "densenet121":
            args = args_densenet121
        elif arch == "vgg16":
            args = args_vgg16
        elif arch == "vgg19":
            args = args_vgg19
        else:
            args = args_preactresnet18
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data

        train_loader, test_loader = load_cifar10(args)
        # Model
        print(f"Train models: {arch}")
        model = load_model(arch)
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in tqdm(range(args['epochs'])):

            train_loss, train_acc = train(train_loader, model, optimizer)
            print(args)
            print('acc: {}'.format(train_acc))

            test_loss, test_acc = test(test_loader, model)

            # save model
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()

        print('Best acc:')
        print(best_acc)


def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, targets = inputs.cuda(), soft_labels.cuda()
        # targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg

def test(testloader, model):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    for (inputs, soft_labels) in testloader:
        inputs, targets = inputs.cuda(), soft_labels.cuda()
        # targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        acc = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg


def save_checkpoint(state, arch):
    filepath = os.path.join("cifar10_models/checkpoints/" + arch + '.pth.tar')
    torch.save(state, filepath)

if __name__ == '__main__':
    main()