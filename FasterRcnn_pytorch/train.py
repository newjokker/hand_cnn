# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import os
import cv2
import datetime
import sys
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
    * python 3.6.10 
    * torch-1.5.0+cu101-cp36-cp36m-linux_x86_64.whl
    * torchvision-0.6.0+cu101-cp36-cp36m-linux_x86_64.whl
# 当前代码验证部分是 cpu 跑的，所以会特别慢
# 参考：
"""

# fixme 只保存效果最好的模型
# fixme 更改验证代码，现在验证代码太慢了 10 张图片运算要 900+ s
# todo 实现裁剪 transform
# todo 增加训练日志，记录每一次训练使用的语句

"""
* python3 train.py -rd /home/ldq/000_train_data/wtx_fas_train_data -gpu 2 -sf ./model -ep 300 -bs 5 -se 5 -mv 4 
"""


def args_parse():
    """参数解析"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-rd", "--root_dir", type=str)
    ap.add_argument("-gpu", "--gpuID", type=str, default="2", help="")
    ap.add_argument("-sd", "--save_dir", type=str, default="./models", help="")
    ap.add_argument("-sn", "--save_name", type=str, default=None, help="")
    ap.add_argument("-ep", "--epoch_num", type=int, default=300, help="")
    ap.add_argument("-bs", "--batch_size", type=int, default=5, help="")
    ap.add_argument("-am", "--assign_model", type=str, default=None, help="")
    ap.add_argument("-nw", "--num_workers", type=int, default=12, help="")
    ap.add_argument("-se", "--save_epoch", type=int, default=5, help="多少个 epoch 保存一次")
    ap.add_argument("-ae", "--add_epoch", type=int, default=0, help="增加的 epoch")
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

def print_log(metric_logger):
    """打印训练信息"""
    lr = metric_logger.meters['lr']
    loss = metric_logger.meters['loss']
    loss_classifier = metric_logger.meters['loss_classifier']
    loss_box_reg = metric_logger.meters['loss_box_reg']
    loss_objectness = metric_logger.meters['loss_objectness']
    loss_rpn_box_reg = metric_logger.meters['loss_rpn_box_reg']
    #
    print("lr : {0}".format(lr))
    print("loss : {0}".format(loss))
    print("loss_classifier : {0}".format(loss_classifier))
    print("loss_box_reg : {0}".format(loss_box_reg))
    print("loss_objectness : {0}".format(loss_objectness))
    print("loss_rpn_box_reg : {0}".format(loss_rpn_box_reg))
    print('-' * 50)

def save_train_log(train_log_folder):
    """记录训练命令"""
    if not os.path.exists(train_log_folder): os.makedirs(train_log_folder)
    train_log_path = os.path.join(train_log_folder, 'train_log.txt')
    time_str = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(hours=13), "%Y-%m-%d-%H:%M:%S")
    with open(train_log_path, 'a') as txt_file:
        txt_file.write(time_str + " : ")
        txt_file.write(" ".join(sys.argv))
        txt_file.write("\n")


if __name__ == "__main__":

    args = args_parse()
    train_log_dir = "./logs"
    save_train_log(train_log_dir)
    # ----------------------------------------------------------------------------------------------------------------------
    root_dir = args["root_dir"].strip('/')
    device = torch.device('cuda')
    batch_size = args["batch_size"]
    num_epochs = args["epoch_num"]
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpuID"]
    # fixme num_works 多了之后就会报错，显示为内存不足，看看原因
    num_workers = args["num_workers"]
    save_dir = args["save_dir"]
    save_name = args["save_name"]
    save_epoch = args["save_epoch"]
    if save_name is None:
        save_name = os.path.split(root_dir)[1]
    # ----------------------------------------------------------------------------------------------------------------------
    # label_list = ["middle_pole", "single"]
    # label_list = ["ls_lm"]
    # label_list = ["jyzm", "jyzt", 'wtx', "other9"]
    label_list = ["fzc_yt", "fzc_sm", "fzc_gt", "fzc_other", "zd_yt", 'zd_sm', "zd_gt", "zd_other", "qx_yt", "qx_sm", "qx_gt", "other"]
    # ----------------------------------------------------------------------------------------------------------------------
    label_dict = {label_list[i]: i + 1 for i in range(len(label_list))}
    num_classes = len(label_list) + 1
    # ----------------------------------------------------------------------------------------------------------------------

    # get dataset
    train_dataset = GetDataset(root_dir, label_dict, get_transform(train=True))
    dataset_test = GetDataset(root_dir, label_dict, get_transform(train=False))

    # fixme 这边应该直接改为一定的比例进行训练，而不是多少个
    # get train_dataset, test_dataset
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    # get data_loader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,  num_workers=num_workers, collate_fn=utils.collate_fn)

    # get model
    add_epoch = 0
    if args["assign_model"] is None:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    else:
        model = torch.load(args["assign_model"])
        add_epoch = args["add_epoch"]

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

    # learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # training
    for epoch in range(num_epochs):
        # update epoch
        epoch += add_epoch + 1
        # train for one epoch
        # fixme 这边其实返回了一个类似于日志的东西，看一下其中的内容，并保存为日志文件
        # print_freq = 50, 每 50 次进行一次打印
        each_metric_logger = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50)
        # print learning info
        # print_log(each_metric_logger)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #if evaluate % 20 ==0:
        # evaluate(model, data_loader_test, device=device)
        # save model
        if epoch % save_epoch == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, "{0}_{1}_{2}.pth".format(save_name, epoch, epoch*len(data_loader_train)))
            torch.save(model, model_path)


































