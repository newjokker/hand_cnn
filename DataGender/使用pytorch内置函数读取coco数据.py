# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torchvision

img_dir = r""
anno_file_path = r""

# 解析 coco 数据
a = torchvision.datasets.CocoDetection(img_dir, anno_file_path)

# 里面的每一个元素的数据结构是 : [PIL.Image.Image, [{'id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation'}]]
#







