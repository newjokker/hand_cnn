import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):

    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        #
        self.classes = {}
        self.labels = {}
        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        # 这边可以传入变换函数
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        """读取图像"""
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


class Resizer(object):
    """Convert ndarrays in sample to Tensors. 样本中的 ndarrary 转为 tensor，并转为指定大小的正方形"""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        # 这边想把图像转为一个正方形，需要看那条边比较长
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size
        # 使用插值进行 resize
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        # 将图像画到一个空的正方形矩阵中
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        # 将 annot 中的前 4 个值，乘以 比例系数
        annots[:, :4] *= scale
        # 转为 tensor
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors. 有一定的概率图像进行镜像处理"""

    def __call__(self, sample, flip_x=0.5):
        # 指定的概率进入
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            # 图像进行镜像
            image = image[:, ::-1, :]
            # 将镜像后的 annot 调整过来
            rows, cols, channels = image.shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample = {'img': image, 'annot': annots}
        return sample


class Normalizer(object):
    """图像进行标准化处理"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


def collater(data):
    """将数据进行组合，并将矩阵转为需要的 shape """
    # img, annot, scale 转为列表形式
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    # 沿着新的轴连接数组序列
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    # 找到图像中标注最多的一张图，标注的个数
    max_num_annots = max(annot.shape[0] for annot in annots)

    # 图像对应的 annot 进行组合
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        #
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    # 图像 channel 之间进行置换
    # input_patch are <batch size, image height, image width, image channel>
    # in_img will be <batch size, image channel, image height, image width>
    imgs = imgs.permute(0, 3, 1, 2)
    return {'img': imgs, 'annot': annot_padded, 'scale': scales}
