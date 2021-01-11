import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, config, root_dir, split='train',
                 is_train=True, transforms=None):
        self.config = config
        self.root = root_dir
        self.imgset = split
        self.train = is_train
        self.transforms = transforms() if transforms is not None else None

        self._annopath = os.path.join(self.root, "Annotations", "{}.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "{}.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "{}.txt")

        with open(self._imgsetpath.format(self.imgset)) as f:
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]

        self.name2id = dict(zip(self.config.class_names, range(len(self.config.class_names))))
        self.id2name = {v: k for k, v in self.name2id.items()}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        if True:  # not self.train or random.random() <= 0.5:
            img_id, img, boxes, classes = self.load_img_and_boxes(index)

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': img,
                    'bboxes': boxes,
                    'labels': classes
                })

                if len(sample['bboxes']) > 0:
                    img = sample['image']
                    boxes = sample['bboxes']
                    classes = sample['labels']
                    break

        if type(img) is np.ndarray:
            img = torch.from_numpy(img) 
        boxes = torch.Tensor(boxes)
        classes = torch.LongTensor(classes)

        return img_id, img, boxes, classes

    def load_img_and_boxes(self, index):
        def get_xml_label(xml_path):
            root = ET.parse(xml_path).getroot()
            boxes = []
            classes = []

            for obj in root.iter("object"):
                label = obj.find("name").text.strip()
                classes.append(self.name2id[label])

                box = []
                _box = obj.find("bndbox")
                for pos in ["xmin", "ymin", "xmax", "ymax"]:
                    box.append(float(_box.find(pos).text))
                boxes.append(box)

            boxes = np.array(boxes, dtype=np.float32)
            classes = np.array(classes, dtype=np.int64)
            return boxes, classes

        img_id = self.img_ids[index]
        img_path = self._imgpath.format(img_id)
        xml_path = self._annopath.format(img_id)

        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
        except:
            print(img_path)
        boxes, classes = get_xml_label(xml_path)
        return img_id, img, boxes, classes

    def collate_fn(self, data):
        img_ids, imgs_list, boxes_list, classes_list = zip(*data)
        '''
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)

        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        max_num = np.array([box.shape[0] for box in boxes_list]).max()
        max_h = np.array([int(s.shape[1]) for s in imgs_list]).max()
        max_w = np.array([int(s.shape[2]) for s in imgs_list]).max()

        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(
                torch.nn.functional.pad(img, [0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])], value=0.))
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], [0, 0, 0, max_num - boxes_list[i].shape[0]], value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], [0, max_num - classes_list[i].shape[0]], value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        return batch_imgs, batch_boxes, batch_classes
        '''
        return img_ids, imgs_list, boxes_list, classes_list
