# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import math
import sys
import time
import torch
from  . import utils
import torchvision.models.detection.mask_rcnn
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator

from JoTools.detectionResult import DeteObj, OperateDeteRes, DeteRes


def target_to_deteres(target, is_tensor=True):
    """将 target 转为 deteres"""
    if is_tensor:
        boxes = target['boxes'].cpu().data.numpy().tolist()
        labels = target['labels'].cpu().data.numpy().tolist()
        # 处理没有 score 的情况
        if 'scores' in target:
            scores = target['scores'].cpu().data.numpy().tolist()
        else:
            scores = [-1] * len(boxes)
    else:
        boxes = target['boxes']
        labels = target['labels']
        if 'scores' in target:
            scores = target['scores']
        else:
            scores = [-1] * len(boxes)
    # 得到的结果
    dete_res = DeteRes()
    for i in range(len(boxes)):
        each_box = boxes[i]
        each_label = labels[i]
        each_score = scores[i]
        x1, y1, x2, y2 = each_box
        dete_res.add_obj(x1, y1, x2, y2, tag=each_label, conf=each_score)
    return dete_res


"""
target 是一个 list 其中的每一个元素是字典，具体 type 如下:

{'boxes': tensor([[586.9481, 259.2025, 866.3922, 319.0185],
        [163.7322, 384.6138, 181.3783, 408.0201],
        [236.9199, 416.9780, 264.4015, 438.3615],
        [159.9716, 446.7415, 178.1962, 466.6802],
        [474.9970, 336.3565, 498.7179, 360.6296],
        [223.0345, 369.2986, 247.9126, 387.7924],
        [288.7010, 375.9449, 315.8932, 403.3966],
        [365.6493, 358.3179, 390.8166, 386.6366],
        [108.4798, 444.7188, 127.5723, 462.9236],
        [150.1361, 421.0236, 169.2285, 434.3160],
        [203.9421, 366.1200, 222.7452, 377.9676]], device='cuda:0'), 'labels': tensor([13, 17, 17, 17, 12, 16, 16, 16, 16, 16, 16], device='cuda:0'), 'image_id': tensor([1544], device='cuda:0'), 'area': tensor([199962.,   4941.,   7030.,   4347.,   6888.,   5504.,   8930.,   8526.,
          4158.,   3036.,   2665.], device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')}
"""


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """训练一个 epoch，返回训练信息"""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    index = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # fixme 打印 loss
        index += 1
        # print("{0} loss : ".format(index), losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def dict_add(dict_1, dict_2):
    """字典之间的相加"""
    res = dict_1.copy()
    # 整合
    for each_key in dict_2:
        if each_key in res:
            res[each_key] += dict_2[each_key]
        else:
            res[each_key] = dict_2[each_key]
    return res


@torch.no_grad()
def evaluate(model, data_loader, device, conf=0.5):
    cpu_device = torch.device("cuda")
    model.eval()

    a = OperateDeteRes()
    a.label_list = [1,2,3]
    res_dict = {}

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        # 参考 ：https://blog.csdn.net/u013548568/article/details/81368019
        torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        for i in range(len(outputs)):
            res = target_to_deteres(outputs[i])
            real = target_to_deteres(targets[i])
            res.filter_by_tages(need_tag=[1,2,3])
            real.filter_by_tages(need_tag=[1,2,3])
            res.filter_by_conf(conf)
            rere = a.compare_customer_and_standard(real, res)
            res_dict = dict_add(res_dict, rere)    
    return res_dict 
