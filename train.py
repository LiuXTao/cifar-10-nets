from models.AlexNet import AlexNet
import torch
from models.ResNet import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

EPOCH = 50  # number of times for each run-through
BATCH_SIZE = 100  # number of images for each epoch
ACCURACY = 0  # overall prediction accuracy
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 10 classes containing in CIFAR-10 dataset
best_loss = None
best_acc = None
best_epoch = 0
parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch',default=EPOCH, type=int, help='number of epochs tp train for')
parser.add_argument('--trainBatchSize', default=BATCH_SIZE, type=int, help='testing batch size')
parser.add_argument('--testBatchSize', default=BATCH_SIZE, type=int, help='testing batch size')
args = parser.parse_args()

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])  # dataset training transform
test_transform = transforms.Compose([transforms.ToTensor()])     # dataset testing transform

def get_data():
    print("###############prepare data########################")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.trainBatchSize, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.testBatchSize, shuffle=True)
    print('#############prepare data finished#################')
    return train_loader, test_loader

def test(epoch):
    # alexnet.eval()
    resnet.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        outs = resnet(data)
        test_loss += F.cross_entropy(outs, target, size_average=False).data[0]
        pred = outs.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return test_loss, accuracy

def train():
    print("###################training########################")
    for epoch in range(args.epoch):
        resnet.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            outs = resnet(data)
            loss = loss_function(outs, target)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            _, pred = outs.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
            accuracy = 100.*correct/total

            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), train_loss/(batch_idx+1), accuracy))

        scheduler.step()
        loss, accuracy = test(epoch)
        t_best_loss, t_best_acc, t_best_epoch = save_model(epoch, accuracy, loss, best_acc, best_loss)
        print('\nCurrent Best epoch {}: Best loss: {:.4f}, Best Accuracy: ({:.2f}%)\n'.format(
            t_best_epoch, t_best_loss, t_best_acc))
    print("################training finished########################")

def save_model(epoch, accuracy, loss, best_acc, best_loss):
    if best_loss == None:
        best_epoch = epoch
        best_loss = loss
        best_acc = accuracy
        file = 'saved_models/best_save_resnet_model.p'
        torch.save(resnet.state_dict(), file)
    elif loss<best_loss and accuracy>best_acc:
        best_epoch = epoch
        best_loss = loss
        best_acc = accuracy
        file = 'saved_models/best_save_resnet_model.p'
        torch.save(resnet.state_dict(), file)
    return best_loss, best_acc, best_epoch

if __name__ == '__main__':
    print("###############prepare model#######################")
    # alexnet = AlexNet()
    resnet = resnet18()
    optimizer = optim.Adam(resnet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)  # lr decay
    loss_function = nn.CrossEntropyLoss()
    print("###########prepare model finished##################")
    train_loader, test_loader = get_data()
    train()
