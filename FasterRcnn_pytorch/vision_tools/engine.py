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

# 参考 :  https://github.com/pytorch/vision/tree/master/references/detection

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
            # fixme 这边是不是需要退出，不能 continue 吗
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


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # cpu_device = torch.device("cpu")
    cpu_device = torch.device("cuda")
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
