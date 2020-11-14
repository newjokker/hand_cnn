# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torch
import os
import cv2
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

# fixme 参数设置为传参模式
# todo 除了 test 和 train 将其他 py 放到辅助文件夹中去
# fixme 更改为断点继续训练
# fixme 只保存效果最好的模型

# ----------------------------------------------------------------------------------------------------------------------
root_dir = r'/home/ldq/000_train_data/wtx_fas_train_data'
# 3 classes, mark_type_1，mark_type_2，background
num_classes = 5
# train on the GPU or on the CPU, if a GPU is not available
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda')
batch_size = 5
# let's train it for   epochs
num_epochs = 300
#
num_workers = 12
# 指定使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
label_list = ["jyzm", "jyzt", 'wtx', "other9"]
label_dict = { label_list[i]:i+1 for i in range(len(label_list))}

# ----------------------------------------------------------------------------------------------------------------------


def get_transform(train):
    # converts the image, a PIL image, into a PyTorch Tensor
    assign_transforms = [T.ToTensor()]
    # assign_transforms.append(T.ToTensor())

    if train:
        # 水平旋转
        assign_transforms.append(T.RandomHorizontalFlip(0.5))
        # 改变图像亮度
        assign_transforms.append(T.RandomChangeImgLight(0.5))
        # 增加噪声
        # assign_transforms.append(T.AddGasussNoise(0.5))
        # 改变通道顺序
        # assign_transforms.append(T.RandomChangechannelOrder(0.5))

    #

    return T.Compose(assign_transforms)


# def xml_info_to_target(xml_info):
#     """将 xml_info 转为 target 格式"""
#     boxes, labels = [], []
#     for each_object in xml_info['object']:
#         name = each_object['name']
#         labels.append(np.int(label_dict[name]))
#         bndbox = each_object['bndbox']
#         xmin, ymin, xmax, ymax = float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(
#             bndbox['ymax'])
#         boxes.append([xmin, ymin, xmax, ymax])
#
#     boxes = torch.as_tensor(boxes, dtype=torch.float32)
#     labels = torch.as_tensor(labels, dtype=torch.int64)
#     image_id = torch.tensor([idx])
#     area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#     # suppose all instances are not crowd
#     iscrowd = torch.zeros((len(xml_info['object']),), dtype=torch.int64)
#     #
#     target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
#     return target
#
#
# def target_to_xml_info(target):
#     """将 target 转为 xml_info 样式"""
#     pass
#
#
# class MarkDataset(torch.utils.data.Dataset):
#     """解析数据，得到符合规范的 dataset"""
#
#     def __init__(self, root, assign_transforms=None):
#         self.root = root
#         self.transforms = assign_transforms
#         # fixme 如果两个文件夹中的文件不一样多就会出现问题了，所以这个逻辑是不是需要改一下
#         # load all image files, sorting them to ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
#         self.xmls = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
#
#     def __getitem__(self, idx):
#         # load images and bbox
#         img_path = os.path.join(root_dir, "JPEGImages", self.imgs[idx])
#         xml_path = os.path.join(root_dir, "Annotations", self.xmls[idx])
#         #
#         img = Image.open(img_path).convert("RGB")
#         # 读取 xml 信息
#         xml_info = parse_xml(xml_path)
#         target = xml_info_to_target(xml_info)
#
#         # boxes, labels = [], []
#         # for each_object in xml_info['object']:
#         #     name = each_object['name']
#         #     labels.append(np.int(label_dict[name]))
#         #     bndbox = each_object['bndbox']
#         #     xmin, ymin, xmax, ymax = float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(
#         #         bndbox['ymax'])
#         #     boxes.append([xmin, ymin, xmax, ymax])
#         #
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # labels = torch.as_tensor(labels, dtype=torch.int64)
#         # image_id = torch.tensor([idx])
#         # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # # suppose all instances are not crowd
#         # iscrowd = torch.zeros((len(xml_info['object']),), dtype=torch.int64)
#         # #
#         # target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
#
#         if self.transforms is not None:
#             # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
#             # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
#             img, target = self.transforms(img, target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)



if __name__ == "__main__":

    # use our dataset and defined transformations
    # todo 修改这边的 get_transform，增加一些变换进去
    dataset = GetDataset(root_dir, label_dict, get_transform(train=True))
    dataset_test = GetDataset(root_dir, label_dict, get_transform(train=False))


    # fixme 将数据集分为 训练集和验证集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,  num_workers=4, collate_fn=utils.collate_fn)

    # get the model using our helper function get_object_detection_model(num_classes)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

    # 学习率管理器
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
        evaluate(model, data_loader_test, device=device)
        # save model
        if epoch % 5 == 0:
            if not os.path.exists(r"./models"):
                os.makedirs(r"./models")
            model_path = r"./models/diy_fas_{0}.pth".format(epoch)
            torch.save(model, model_path)

    print("That's it!")


































