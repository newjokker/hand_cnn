import os
import math
import time
import random
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

from config import DefaultConfig
from transforms import train_transforms, valid_transforms
from my_dataset import VOCDataset
from eval import *


def train(opt, config):
    model = fasterrcnn_resnet50_fpn(pretrained=True,
                                    rpn_pre_nms_top_n_train=config.rpn_pre_nms_top_n_train,
                                    rpn_post_nms_top_n_train=config.rpn_post_nms_top_n_train,
                                    rpn_pre_nms_top_n_test=config.rpn_pre_nms_top_n_test,
                                    rpn_post_nms_top_n_test =config.rpn_post_nms_top_n_test
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.class_num)
    model.to(device)
    #model = torch.nn.DataParallel(model)
    model.train()
    print(model)

    train_dataset = VOCDataset(config, root_dir=config.trainval_dir, split="trainval",
                               transforms=train_transforms, is_train=True)
    valid_dataset = VOCDataset(config, root_dir=config.trainval_dir, split="test",
                               transforms=valid_transforms, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=opt.n_cpu,
                              worker_init_fn=np.random.seed(config.SEED))
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=train_dataset.collate_fn, num_workers=opt.n_cpu,
                              worker_init_fn=np.random.seed(config.SEED))
    print("total_images : {}".format(len(train_dataset)))

    EPOCHS = config.EPOCHS
    STEPS_PER_EPOCH = math.ceil(len(train_dataset) / config.BATCH_SIZE)
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMPUP_STEPS = 501
    GLOBAL_STEPS = 1
    optimizer = config.optimizer_class(model.parameters(), **config.optimizer_params)

    folder = time.strftime("%Y%m%d__%H_%M", time.localtime())
    save_dir = os.path.join(config.exp_dir, folder, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best = 0.

    print("=" * 20, "start training", "=" * 20)
    for epoch in range(EPOCHS):
        epoch_start_time = time.strftime("%Y%m%d %H:%M", time.localtime())
        print("\r\nEPOCH: {}, start: {}".format(epoch+1, epoch_start_time))
        
        model.train()
        total_loss = 0
        for epoch_step, data in enumerate(train_loader):
            img_ids, batch_imgs, batch_boxes, batch_classes = data
            inputs = list(img.type(torch.FloatTensor).to(device) for img in batch_imgs)

            targets = []
            for i in range(len(batch_boxes)):
                target = {}
                target["boxes"] = batch_boxes[i].type(torch.FloatTensor).to(device)
                target["labels"] = batch_classes[i].to(device)
                targets.append(target)

            # warm up
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * config.LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            # lr decay
            elif GLOBAL_STEPS == int(TOTAL_STEPS*0.667):
                lr = config.LR_INIT * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr
            elif GLOBAL_STEPS == int(TOTAL_STEPS*0.889):
                lr = config.LR_INIT * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr

            optimizer.zero_grad()
            loss_dict = model(inputs, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss
            loss.mean().backward()
            optimizer.step()

            if epoch_step % config.verbose_interval == 0:
                print('step: {:4d}/{}, lr: {:.7f}, '.format(epoch_step+1, STEPS_PER_EPOCH, lr) +
                      'rpn_cls: {:.5f}, rpn_reg: {:.5f}, det_cls: {:.5f}, det_reg: {:.5f}, total: {:.5f}'.format(
                      loss_dict["loss_objectness"], loss_dict["loss_rpn_box_reg"], loss_dict["loss_classifier"], loss_dict["loss_box_reg"], total_loss/(epoch_step+1)),
                      end='\r')
            GLOBAL_STEPS += 1
        model_name = "model_{}.pth".format(epoch + 1)
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))

        model.eval()
        print("\r\n=== eval ===")
        gt_boxes = []
        gt_classes = []
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        total_val_epochs = math.ceil(len(valid_dataset) / config.BATCH_SIZE)
        for val_epoch, data in enumerate(valid_loader):
            img_ids, batch_imgs, batch_boxes, batch_classes = data
            inputs = list(img.type(torch.FloatTensor).to(device) for img in batch_imgs)

            with torch.no_grad():
                pred = model(inputs)
            for i in range(len(pred)):
                tmp = pred[i]["boxes"].cpu()
                pred_boxes.append(tmp)
                tmp = pred[i]["labels"].cpu()
                pred_classes.append(tmp)
                tmp = pred[i]["scores"].cpu()
                pred_scores.append(tmp)

                gt_boxes.append(batch_boxes[i])
                gt_classes.append(batch_classes[i])
            del inputs
            del pred

            print("{} / {}".format(val_epoch+1, total_val_epochs), end='\r')

        pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
        label_APs = eval_ap_2d(gt_boxes, gt_classes,pred_boxes, pred_classes, pred_scores,0.5, config.eval_labels)
        mAP = 0.
        for label_mAP in label_APs.values():
            mAP += float(label_mAP)
        mAP /= len(config.eval_labels)
        print("mAP: {}".format(mAP))
        if mAP > best:
            best = mAP
            model_name = "model_best.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))


def fix_seed(config):
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="gpu during training")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_config = DefaultConfig
    fix_seed(train_config)

    train(args, train_config)













