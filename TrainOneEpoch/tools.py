# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torchvision
import torch
# import transforms as T
from pycocotools import mask as coco_mask
import numpy as np
import transforms as T



def collate_fn(batch):
    """用于将一个 batch 中的数据进行组合"""
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        """这个函数的主要作用就是进行一次 transform ，并将 target 的格式进行变换为需要的样式"""
        img, target = super(CocoDetection, self).__getitem__(idx)

        # img = torch.from_numpy(np.array(img))
        # img = img.permute(2, 0, 1)

        image_id = self.ids[idx]
        new_target = []

        for each in target:
            each_target = {'image_id': torch.tensor(each['image_id']),
                           'iscrowd': torch.tensor(each['iscrowd'], dtype=torch.int64),        # torch.zeros((num_objs,), dtype=torch.int64)
                           'area': torch.tensor(each['area'], dtype=torch.float32),
                           'boxes': torch.as_tensor(each['bbox'], dtype=torch.float32),
                           'labels': torch.tensor(each['category_id'], dtype=torch.int64)}
            new_target.append(each_target)

        # fixme target 的数据结构比较繁琐 [PIL.Image.Image, [{'image_id', 'iscrowd', 'area', 'boxes', 'labels'}]]
        # target = dict(image_id=image_id, annotations=target)
        target = dict(image_id=image_id, annotations=new_target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def get_data_loader():
    """获取一个 data_loader"""

    train_img_dir = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets/fzc_broken_class/train"
    anno_file_path = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets/fzc_broken_class/annotations/instances_train.json"

    train_img_dir = r"C:\Users\14271\Desktop\优化开口销第二步\013_整理efficientdet格式的螺母缺失数据\ls_lm_eff_train_data\train"
    anno_file_path = r"C:\Users\14271\Desktop\优化开口销第二步\013_整理efficientdet格式的螺母缺失数据\ls_lm_eff_train_data/annotations/instances_train.json"


    batch_size = 10
    num_workers = 0

    # train_dataset = torchvision.datasets.CocoDetection(train_img_dir, anno_file_path)
    train_dataset = CocoDetection(train_img_dir, anno_file_path, transforms=get_transform('train'))

    # 获得 batch_sampler
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    # 获得 data_loader
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader

