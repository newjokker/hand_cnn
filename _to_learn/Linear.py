# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
from torch import nn

"""
* 全连接层（Fully Connected Layer）
"""


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # requires_grad=True : cal grad, https://www.cnblogs.com/dychen/p/13921791.html
        self.w = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.randn(out_dim), requires_grad=True)

    def forward(self, x):
        x = x.matmul(self.w)
        y = x + self.b.expand_as(x)
        return y


class Perception(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_dim, hid_dim)
        self.layer2 = Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        y = torch.sigmoid(x)
        y = self.layer2(y)
        y = torch.sigmoid(y)
        return y


class PerceptionNew(nn.Module):
    """ test Sequential """

    def __init__(self, in_dim, hid_dim, out_dim):
        super(PerceptionNew, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.Sigmoid(), nn.Linear(hid_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        y = self.layer(x)
        return y



if __name__ == "__main__":

    a = PerceptionNew(2, 5, 3)
    data = torch.randn(4, 2)
    out = a(data)
    print(a)
    print(data)
    print(out)






