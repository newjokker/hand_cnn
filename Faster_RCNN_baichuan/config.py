import torch.optim as optim


class DefaultConfig:
    # path
    pretrained = True
    trainval_dir = "data/VOC2007"
    test_dir = "/home/fbc/Datasets/VOC/VOCdevkit/VOC2007"
    exp_dir = "exps"

    # dataset
    class_names = ["__background__", "L88", "tk", "LX", "LSJ", "td", "fn", "jyz", "yx", "nc", "cc", "kkx2", "tower", "car", "nest"]
    class_num = len(class_names) 
    class2ind = dict(zip(class_names, range(len(class_names))))
    eval_labels = ["nc"]
    for i, label in enumerate(eval_labels):
        eval_labels[i] = class2ind[label]

    # rpn
    rpn_pre_nms_top_n_train= 12000
    rpn_post_nms_top_n_train= 2000
    rpn_pre_nms_top_n_test= 6000
    rpn_post_nms_top_n_test= 1000

    # train
    SEED = 0
    BATCH_SIZE = 4
    EPOCHS = 24
    LR_INIT = 1e-4
    optimizer_class = optim.SGD
    optimizer_params = dict(lr=LR_INIT, momentum=0.9, weight_decay=1e-4)
    verbose_interval = 1

    # inference
    score_threshold = 0.05
    nms_iou_threshold = 0.3
    max_detection_boxes_num = 150
