import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
from dataset.coco_utils import get_coco_api_from_dataset
from dataset.coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """一个 epoch 的训练"""

    # fixme 模型设置为训练模式？
    model.train()
    # 日志文件相关
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    #
    if epoch == 0:
        # 设置初始化的要素？
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        # fixme 这边的学习率好像和外面一层的学习率完全不是一个东西，确定一下是干什么用的
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # print_freq, 打印频率，header 打印时放在头部的字符串
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 将图像和标签，处理为需要的格式，图像读取为四维矩阵，标签存放到 GPU
        # device, 使用的终端，这边是指，cuda，
        images = list(image.to(device) for image in images)
        # 与分类不同，这边的 target 除了有 tag ，还应该有位置，有分类损失 + 回归损失
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # fixme 得到损失值，这边的 target 的结构是什么 ？？？
        # fixme 这边的 model(images, targets) 执行的是 model forward 函数里面的内容，因为这当中写法会执行 __call__ 里面的内容， 复函数 __call__ 里面指定执行  forward 函数


        # --------------
        #  targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        # --------------



        loss_dict = model(images, targets)
        # 直接将损失值进行相加，应该有分类损失和回归损失两种，这边直接进行了相加，对于 kkx_step_2 这边应该设置的是 分类损失的权重占比比较小，回归损失的占比比较大
        losses = sum(loss for loss in loss_dict.values())

        # --------------------------------------------------------------------------------------------------------------
        # fixme 这边看不懂，需要好好看看是干什么用的
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        # --------------------------------------------------------------------------------------------------------------

        # 判断损失值是不是 NAN inf 或者其他无限的值，是的话就停止训练 loss，不正常的原因 ：https://oldpan.me/archives/careful-train-loss-nan-inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 清空数据
        optimizer.zero_grad()
        # 计算梯度
        losses.backward()
        # 更新模型参数
        optimizer.step()

        # 边的策略是一个 batch 更新一次学习率，一个 epoch 初始化一次学习率
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 更新日志文件
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


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
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

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
