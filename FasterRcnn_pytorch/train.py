# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import os
import cv2
import argparse
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from vision_tools import transforms as T
from vision_tools.engine import train_one_epoch, evaluate
from vision_tools import utils
from vision_tools.jo_dataset import GetDataset
from JoTools.txkj.parseXml import parse_xml


"""
# 运行的 torch 环境
    * torch-1.5.0+cu101-cp36-cp36m-linux_x86_64.whl
    * torchvision-0.6.0+cu101-cp36-cp36m-linux_x86_64.whl
# 当前代码验证部分是 cpu 跑的，所以会特别慢
# 参考：
"""

# fixme 更改为断点继续训练
# fixme 只保存效果最好的模型

"""
* python3 train.py -rd /home/ldq/000_train_data/wtx_fas_train_data -gpu 2 -sf ./model -ep 300 -bs 5
"""


def args_parse():
    """参数解析"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-rd", "--root_dir", type=str)
    ap.add_argument("-gpu", "--gpuID", type=str, default="2", help="")
    ap.add_argument("-sf", "--save_folder", type=str, default="./res", help="")
    ap.add_argument("-sn", "--save_name", type=str, default=None, help="")
    ap.add_argument("-ep", "--epoch_num", type=int, default=300, help="")
    ap.add_argument("-bs", "--batch_size", type=int, default=5, help="")
    assign_args = vars(ap.parse_args())  # vars 返回对象object的属性和属性值的字典对象
    return assign_args

def get_transform(train):
    # converts the image, a PIL image, into a PyTorch Tensor
    assign_transforms = [T.ToTensor()]
    # assign_transforms.append(T.ToTensor())

    if train:
        # 水平旋转
        assign_transforms.append(T.RandomHorizontalFlip(0.5))
        # 改变图像亮度
        assign_transforms.append(T.RandomChangeImgLight(0.5))
        # 改变通道顺序
        assign_transforms.append(T.RandomChangechannelOrder(0.5))
        # 增加噪声
        # assign_transforms.append(T.AddGasussNoise(0.5))

    return T.Compose(assign_transforms)



if __name__ == "__main__":

    args = args_parse()
    # ----------------------------------------------------------------------------------------------------------------------
    root_dir = args["root_dir"]
    device = torch.device('cuda')
    batch_size = args["batch_size"]
    num_epochs = args["epoch_num"]
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpuID"]
    num_workers = 12
    save_dir = args["save_folder"]
    save_name = args["save_name"]
    if save_name is None:
        save_name = os.path.split(save_dir)[1]
    # ----------------------------------------------------------------------------------------------------------------------
    label_list = ["jyzm", "jyzt", 'wtx', "other9"]
    # ----------------------------------------------------------------------------------------------------------------------
    label_dict = {label_list[i]: i + 1 for i in range(len(label_list))}
    num_classes = len(label_list) + 1
    # ----------------------------------------------------------------------------------------------------------------------

    # get dataset
    train_dataset = GetDataset(root_dir, label_dict, get_transform(train=True))
    dataset_test = GetDataset(root_dir, label_dict, get_transform(train=False))

    # get train_dataset, test_dataset
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    # get data_loader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,  num_workers=num_workers, collate_fn=utils.collate_fn)

    # get model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

    # learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # training
    for epoch in range(num_epochs):
        # train for one epoch
        # fixme 这边其实返回了一个类似于日志的东西，看一下其中的内容，并保存为日志文件
        # print_freq = 50, 每 50 次进行一次打印
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if evaluate % 20 ==0:
            evaluate(model, data_loader_test, device=device)
        # save model
        if epoch % 5 == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, "{0}_{1}_{2}.pth".format(save_name, epoch, epoch*len(data_loader_train)))
            torch.save(model, model_path)


































