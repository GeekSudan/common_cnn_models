""" helper function

author baiyu
"""

import sys

import numpy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from conf import settings

LEARNING_RATE = 0.01

WEIGHT_DECAY = 0.0001

MOMENTUM = 0.9

I = 3
I = float(I)


# from dataset import CIFAR100Train, CIFAR100Test

def get_optimizer(parameters, optimizer):
    if optimizer == 'znd':
        from znd import ZNN2Optimizer
        optimizer = ZNN2Optimizer(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, I=I)
    elif optimizer == 'znd_random':
        from random_noise.znd_random_noise import ZNNRandomOptimizer
        optimizer = ZNNRandomOptimizer(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, I=I)
    elif optimizer == 'znd_constant':
        from constant_noise.znd_constant_noise import ZNNConstant
        optimizer = ZNNConstant(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, I=I)
    elif optimizer == 'momentum':
        from torch.optim.sgd import SGD
        optimizer = SGD(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif optimizer == 'momentum_random':
        from random_noise.momentum_random_noise import MomentumRandom
        optimizer = MomentumRandom(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif optimizer == 'momentum_constant':
        from constant_noise.momentum_constant_noise import MomentumConstant
        optimizer = MomentumConstant(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif optimizer == 'adam':
        from torch.optim.adam import Adam
        optimizer = Adam(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    elif optimizer == 'adam_random':
        from random_noise.adam_random_noise import AdamRandom
        optimizer = AdamRandom(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    elif optimizer == 'adam_constant':
        from constant_noise.adam_constant_noise import AdamConstant
        optimizer = AdamConstant(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    else:
        print('the optimizer name you have entered is not supported yet')
        sys.exit()

    return optimizer


def get_network(network, use_gpu=True):
    """ return given network
    """

    if network == 'vgg16':
        from model.vgg import vgg16_bn
        net = vgg16_bn()
    elif network == 'vgg13':
        from model.vgg import vgg13_bn
        net = vgg13_bn()
    elif network == 'vgg11':
        from model.vgg import vgg11_bn
        net = vgg11_bn()
    elif network == 'vgg19':
        from model.vgg import vgg19_bn
        net = vgg19_bn()
    elif network == 'densenet121':
        from model.densenet import densenet121
        net = densenet121()
    elif network == 'densenet161':
        from model.densenet import densenet161
        net = densenet161()
    elif network == 'densenet169':
        from model.densenet import densenet169
        net = densenet169()
    elif network == 'densenet201':
        from model.densenet import densenet201
        net = densenet201()
    elif network == 'googlenet':
        from model.googlenet import googlenet
        net = googlenet()
    elif network == 'inceptionv3':
        from model.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif network == 'inceptionv4':
        from model.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif network == 'inceptionresnetv2':
        from model.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif network == 'xception':
        from model.xception import xception
        net = xception()
    elif network == 'resnet18':
        from model.resnet import resnet18
        net = resnet18()
    elif network == 'resnet34':
        from model.resnet import resnet34
        net = resnet34()
    elif network == 'resnet50':
        from model.resnet import resnet50
        net = resnet50()
    elif network == 'resnet101':
        from model.resnet import resnet101
        net = resnet101()
    elif network == 'resnet152':
        from model.resnet import resnet152
        net = resnet152()
    elif network == 'preactresnet18':
        from model.preactresnet import preactresnet18
        net = preactresnet18()
    elif network == 'preactresnet34':
        from model.preactresnet import preactresnet34
        net = preactresnet34()
    elif network == 'preactresnet50':
        from model.preactresnet import preactresnet50
        net = preactresnet50()
    elif network == 'preactresnet101':
        from model.preactresnet import preactresnet101
        net = preactresnet101()
    elif network == 'preactresnet152':
        from model.preactresnet import preactresnet152
        net = preactresnet152()
    elif network == 'resnext50':
        from model.resnext import resnext50
        net = resnext50()
    elif network == 'resnext101':
        from model.resnext import resnext101
        net = resnext101()
    elif network == 'resnext152':
        from model.resnext import resnext152
        net = resnext152()
    elif network == 'shufflenet':
        from model.shufflenet import shufflenet
        net = shufflenet()
    elif network == 'shufflenetv2':
        from model.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif network == 'squeezenet':
        from model.squeezenet import squeezenet
        net = squeezenet()
    elif network == 'mobilenet':
        from model.mobilenet import mobilenet
        net = mobilenet()
    elif network == 'mobilenetv2':
        from model.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif network == 'nasnet':
        from model.nasnet import nasnet
        net = nasnet()
    elif network == 'attention56':
        from model.attention import attention56
        net = attention56()
    elif network == 'attention92':
        from model.attention import attention92
        net = attention92()
    elif network == 'seresnet18':
        from model.senet import seresnet18
        net = seresnet18()
    elif network == 'seresnet34':
        from model.senet import seresnet34
        net = seresnet34()
    elif network == 'seresnet50':
        from model.senet import seresnet50
        net = seresnet50()
    elif network == 'seresnet101':
        from model.senet import seresnet101
        net = seresnet101()
    elif network == 'seresnet152':
        from model.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    # if use_gpu:
    #     net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=30, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=30, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

# class WarmUpLR(_LRScheduler):
#     """warmup_training learning rate scheduler
#     Args:
#         optimizer: optimzier(e.g. SGD)
#         total_iters: totoal_iters of warmup phase
#     """
#     def __init__(self, optimizer, total_iters, last_epoch=-1):
#
#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         """we will use the first m batches, and set the learning
#         rate to base_lr * m / total_iters
#         """
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
