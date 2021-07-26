# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# 自定义分类网络
# 手写数字分类网络

# --------------------------------------------------
# todo 如何缩小模型，裁剪模型
# todo 学习率的更新
# --------------------------------------------------

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

# exit()

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        #
        self.BN_20 = nn.BatchNorm2d(20)                     # BatchNorm2d 显著增加了模型的性能
        self.BN_40 = nn.BatchNorm2d(40)
        #
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(640, 10)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        x = self.BN_20(self.mp(self.conv1(x)))
        x = F.relu(x)
        # x: 64*20*4*4
        x = self.BN_40(self.mp(self.conv2(x)))
        x = F.relu(x)
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return F.log_softmax(x)


def train(epoch):
    #
    for batch_idx, (data, target) in enumerate(train_loader):
        # fixme 不明白这边为什么要将 tensor 转为 Variable
        # data, target = Variable(data), Variable(target)
        # 使用当前命名空间中的 grad，所以需要 optimizer 每次清空
        optimizer.zero_grad()
        output = model(data)
        # 计算 loss
        loss = F.nll_loss(output, target)
        # 计算反向梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 日志
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))




if __name__ == "__main__":


    model_path = r"./model/demo.pth"
    # 加载模型
    model = torch.load(model_path)
    # model = Net()
    # 优化器需要和 model 绑定，因为要执行 model 参数的更新
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(3):
        train(epoch)
        test()

    # 保存模型
    torch.save(model, model_path)

