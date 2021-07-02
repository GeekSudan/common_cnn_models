import time

import torch
import torch.nn as nn
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from conf import settings
from util import get_network, get_training_dataloader, get_test_dataloader, get_optimizer

I = 3
I = float(I)

model_save_dir = '/data/mnist/models'

# Hyper Parameters
# input_size = 784
# hidden_size = 1000
# num_classes = 10
num_epochs = 100
batch_size = 30
learning_rate = 0.01

logger = Logger('znd_dens101_random.txt', title='cifar')

logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# data preprocessing:
cifar100_training_loader = get_training_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,

)

cifar100_test_loader = get_test_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,

)

net = get_network('densenet121')
device = torch.device('cuda')
net = net.to(device)
net.train()
# # Loss and Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = get_optimizer(net.parameters(), 'znd_random')
start_time = time.time()
loss_collection = []
episode_no = 0

# Train the Model
iters = 0
for epoch in range(num_epochs):

    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()

    average_loss = 0.0

    for i, (inputs, labels) in enumerate(cifar100_training_loader):
        # clear the parameter gradients
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # collect predictions
        outputs = net.forward(inputs)

        # calculate loss and collect gradients
        train_loss = criterion(outputs, labels)
        train_loss.backward()

        # adjust weights
        optimizer.step()
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        train_loss_log.update(train_loss.item(), inputs.size(0))
        train_acc_log.update(prec1.item(), inputs.size(0))

        # add statistics to collection for graphing purposes
        # average_loss += train_loss.item()
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                  % (epoch + 1, num_epochs, i + 1, len(cifar100_training_loader) // 16, train_loss_log.avg / i,
                     train_acc_log.avg))

    net.eval()
    correct = 0
    loss = 0
    total = 0
    for images, labels in cifar100_test_loader:
        images, labels = images.to(device), labels.to(device)
        # images, labels = images.cuda(), labels.cuda()
        outputs = net.forward(images)
        test_loss = criterion(outputs, labels)
        val_loss_log.update(test_loss.item(), images.size(0))
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        val_acc_log.update(prec1.item(), images.size(0))

    logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
    print('Accuracy of the network on the 10000 test images: %.8f' % (val_acc_log.avg))
    print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))

logger.close()
print("Elapsed time " + (str(time.time() - start_time)))
logger.plot()
