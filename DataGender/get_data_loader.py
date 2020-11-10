# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torchvision
import torch
# import transforms as T
from pycocotools import mask as coco_mask


# ----------------------------------------------------------------------------------------------------------------------
# train_img_dir = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets/fzc_broken_class/train"
# anno_file_path = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets/fzc_broken_class/annotations/instances_train.json"
# batch_size = 10
# num_workers = 12
# ----------------------------------------------------------------------------------------------------------------------



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        """这个函数的主要作用就是进行一次 transform ，并将 target 的格式进行变换为需要的样式"""
        # fixme 使用父类方法得到 img 和 target
        img, target = super(CocoDetection, self).__getitem__(idx)
        # fixme 这一步的作用没看懂，self.ids 是会打乱排列顺序的吗？
        image_id = self.ids[idx]
        # fixme 对 target 格式进行转换，转为字典格式，有 image_id 和 annotation 两个关键字
        # fixme target 的数据结构比较繁琐 [PIL.Image.Image, [{'id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation'}]]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            # fixme 转换的时候要将 img 传入，因为可能是需要知道原数据信息的操作
            img, target = self._transforms(img, target)
        return img, target


def collate_fn(batch):
    """用于将一个 batch 中的数据进行组合"""
    return tuple(zip(*batch))

def get_data_loader():
    """获取一个 data_loader"""

    # train_img_dir = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets/fzc_broken_class/train"
    # anno_file_path = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets/fzc_broken_class/annotations/instances_train.json"
    train_img_dir = r"C:\Users\14271\Desktop\优化开口销第二步\013_整理efficientdet格式的螺母缺失数据\ls_lm_eff_train_data\train"
    anno_file_path = r"C:\Users\14271\Desktop\优化开口销第二步\013_整理efficientdet格式的螺母缺失数据\ls_lm_eff_train_data/annotations/instances_train.json"


    batch_size = 10
    num_workers = 12

    # train_dataset = torchvision.datasets.CocoDetection(train_img_dir, anno_file_path)
    train_dataset = torchvision.CocoDetection.CocoDetection(train_img_dir, anno_file_path)

    # 获得 batch_sampler
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    # 获得 data_loader
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader


















# 里面的每一个元素的数据结构是 : [PIL.Image.Image, [{'id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation'}]]
#







