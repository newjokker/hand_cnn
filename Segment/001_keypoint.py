# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torchvision
from PIL import Image

model_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model_rcnn.eval()

image1 = Image.open(r'/home/suanfa-4/ldq/001_train_data/fzc_step_1/JPEGImages/202104_352742.jpg')
# image2 = Image.open(r'/home/suanfa-4/ldq/001_train_data/fzc_step_1/JPEGImages/202104_407327.jpg')

image_tensor1 = torchvision.transforms.functional.to_tensor(image1)
# image_tensor2 = torchvision.transforms.functional.to_tensor(image2)

output1 = model_rcnn([image_tensor1])
# output2 = model_rcnn([image_tensor2])


print(output1)
