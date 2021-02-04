# -*- coding: utf-8  -*-
# -*- author: jokker -*-


# todo 完善一个函数，输入 model 和 valid_loader，输出 map



model.eval()
gt_boxes = []
gt_classes = []
pred_boxes = []
pred_classes = []
pred_scores = []
total_val_epochs = math.ceil(len(valid_dataset) / config.BATCH_SIZE)    # 向上取整
#
for val_epoch, data in enumerate(valid_loader):
    img_ids, batch_imgs, batch_boxes, batch_classes = data
    inputs = list(img.type(torch.FloatTensor).to(device) for img in batch_imgs)     # 将图片全部转为 tensor

    # 不跟新参数，进行模型预测
    with torch.no_grad():
        pred = model(inputs)

    # 遍历每一个图片的检测结果
    for i in range(len(pred)):
        tmp = pred[i]["boxes"].cpu()
        pred_boxes.append(tmp)
        tmp = pred[i]["labels"].cpu()
        pred_classes.append(tmp)
        tmp = pred[i]["scores"].cpu()
        pred_scores.append(tmp)
        #
        gt_boxes.append(batch_boxes[i])
        gt_classes.append(batch_classes[i])
    del inputs
    del pred

    print("{} / {}".format(val_epoch + 1, total_val_epochs), end='\r')

#
pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
label_APs = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5, config.eval_labels)
#
mAP = 0.
#
for label_mAP in label_APs.values():
    mAP += float(label_mAP)
#
mAP /= len(config.eval_labels)
print("mAP: {}".format(mAP))
#
if mAP > best:
    best = mAP
    model_name = "model_best.pth"
    torch.save(model.state_dict(), os.path.join(save_dir, model_name))
