# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torch
import torchvision
from tools import get_data_loader


# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
"""一个 epoch 的训练"""


# target 中需要的内容，可以参考这篇文章


# ----------------------------------------------------------------------------------------------------------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=15, pretrained=False)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001)
data_loader = get_data_loader()
device = "cpu"
# ----------------------------------------------------------------------------------------------------------------------

model.train()

print("-"*50)

for images, targets in data_loader:

    # fixme 这边是 image object 需要进行转换，转为 tensor

    print(type(images))

    # images, targets = model.transform(images, targets)
    #
    # # images = list(image.to(device) for image in images)
    # # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #
    # print(11)
    # loss_dict = model(images, targets)
    # print("12")
    # losses = sum(loss for loss in loss_dict.values())
    #
    # optimizer.zero_grad()
    # losses.backward()
    # optimizer.step()

    print("ok")
