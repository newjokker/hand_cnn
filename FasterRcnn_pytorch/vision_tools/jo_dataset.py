# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import torch
import numpy as np
from PIL import Image
from JoTools.txkj.parseXml import parse_xml


def xml_info_to_target(xml_info, label_dict, idx):
    """将 xml_info 转为 target 格式"""
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
    return target


def target_to_xml_info(target):
    """将 target 转为 xml_info 样式"""
    pass


class GetDataset(torch.utils.data.Dataset):
    """解析数据，得到符合规范的 dataset"""

    def __init__(self, root, label_dict, assign_transforms=None):
        self.root_dir = root
        self.label_dict = label_dict
        self.transforms = assign_transforms
        # fixme 如果两个文件夹中的文件不一样多就会出现问题了，所以这个逻辑是不是需要改一下
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root_dir, "JPEGImages", self.imgs[idx])
        xml_path = os.path.join(self.root_dir, "Annotations", self.xmls[idx])
        #
        img = Image.open(img_path).convert("RGB")
        # 读取 xml 信息
        xml_info = parse_xml(xml_path)
        target = xml_info_to_target(xml_info, self.label_dict, idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
