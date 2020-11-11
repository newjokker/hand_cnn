# -*- coding: utf-8  -*-
# -*- author: jokker -*-


from engine import train_one_epoch, evaluate
import utils
import transforms as T
from engine import train_one_epoch, evaluate
import torch
import torchvision
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse
from JoTools.txkj.parseXml import ParseXml, parse_xml

# fixme cpu 训练成功使用的环境是 torch 1.5.0, torchvision 0.6.0
# fixme 如果不是 gpu 版本的 torch 强行使用 gpu 进行训练就可能报错，关于 memory 的
# ----------------------------------------------------------------------------------------------------------------------
root = r'/home/ldq/FasterRcnn/kkx_train_data_2020_10_29'
# 3 classes, mark_type_1，mark_type_2，background
num_classes = 18
# train on the GPU or on the CPU, if a GPU is not available
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
batch_size = 5
model_path = r"./diy_fas.pth"
# ----------------------------------------------------------------------------------------------------------------------


def get_transform(train):
    transforms = [T.ToTensor()]
    # converts the image, a PIL image, into a PyTorch Tensor

    """
    
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))
    """

    return T.Compose(transforms)


class MarkDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # fixme 如果两个文件夹中的文件不一样多就会出现问题了，所以这个逻辑是不是需要改一下
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))[:120]
        self.xmls = list(sorted(os.listdir(os.path.join(root, "Annotations"))))[:120]

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(root, "JPEGImages", self.imgs[idx])
        xml_path = os.path.join(root, "Annotations", self.xmls[idx])
        #
        img = Image.open(img_path).convert("RGB")
        # 读取 xml 信息
        xml_info = parse_xml(xml_path)

        label_dict = {"dense2": 1, "other_L4kkx": 2, 'other_fist': 3, 
                        "K_no_lw": 4, "other2": 5, "other_fzc": 6, "other7": 7, "other8": 8,
                        "other9": 9, "other1": 10, "other6": 11, "K": 12, "dense1": 13,
                        "dense3": 14, "other3": 15, "Lm": 16, "KG": 17,
                        }
        

        boxes, labels = [], []
        for each_object in xml_info['object']:
            name = each_object['name']
            labels.append(np.int(label_dict[name]))
            bndbox = each_object['bndbox']
            xmin, ymin, xmax, ymax = float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(
                bndbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(xml_info['object']),), dtype=torch.int64)
        #
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
            # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# use our dataset and defined transformations
dataset = MarkDataset(root, get_transform(train=True))
dataset_test = MarkDataset(root, get_transform(train=False))


# fixme 这边要进行修改
# fixme 将数据集分为 训练集和验证集
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:100])
dataset_test = torch.utils.data.Subset(dataset_test, indices[100:])

# define training and validation data loaders
# 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=4, collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,  num_workers=4, collate_fn=utils.collate_fn)

# get the model using our helper function
# 或get_object_detection_model(num_classes)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
# cos学习率
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for   epochs
num_epochs = 31

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # engine.py的train_one_epoch函数将images和targets都.to(device)了
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    torch.save(model, model_path)

print("That's it!")

