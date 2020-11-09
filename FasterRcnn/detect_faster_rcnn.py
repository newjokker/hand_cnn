# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torchvision
import cv2
from JoTools.detectionResult import DeteObj, DeteRes

# ----------------------------------------------------------------------------------------------------------------------
model_path = r"/home/ldq/DetectionLearn/Faster-RCNN-with-torchvision/model/model_13.pth"
img_path = r"./img/test.jpg"
# ----------------------------------------------------------------------------------------------------------------------

# 载入自定义训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=91, pretrained=False)
# Moves all model parameters and buffers to the GPU
model = model.cuda()
# Sets the module in evaluation mode # fixme 这个是不是让模型固定权重，不进行训练？
model.eval()

# 得到模型
model_save = torch.load(model_path)
model.load_state_dict(model_save['model'])
src_img = cv2.imread(img_path)
img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
out = model([img_tensor])

# 结果处理并输出
boxes, labels, scores = out[0]['boxes'], out[0]['labels'], out[0]['scores']
#
res = DeteRes(assign_img_path=img_path)
for index, each_box in enumerate(boxes):
    if float(scores[index]) > 0.6:
        x1, y1, x2, y2 = int(each_box[0]), int(each_box[1]), int(each_box[2]), int(each_box[3])
        conf, tag = float(scores[index]), str(labels[index].item())
        res.add_obj(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, tag=tag)
# 保存画图和 xml
res.draw_dete_res(r"./test.jpg")
res.save_to_xml(r"./test.xml")
