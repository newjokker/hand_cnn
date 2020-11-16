# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torchvision
import cv2
import os
import sys
import argparse
from JoTools.detectionResult import DeteObj, DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil



def args_parse():
    """参数解析"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-if", "--img_folder", default=r"../img", help="")
    ap.add_argument("-ip", "--img_path", default=r"../img/test.jpg", help="")
    ap.add_argument("-m", "--model", default=r"./diy_fas_2.pth", help="")
    ap.add_argument("-gpuID", "--gpu_id", type=str, default="2", help="")
    ap.add_argument("-conf", "--conf_th", type=str, default="0.35", help="")
    ap.add_argument("-save", "--save_folder", type=str, default="./res", help="")
    assign_args = vars(ap.parse_args())  # vars 返回对象object的属性和属性值的字典对象
    return assign_args


def dete_one_img(assign_img_path, assign_save_folder):
    """检测一张图片"""
    src_img = cv2.imread(assign_img_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    out = model([img_tensor])

    # 结果处理并输出
    boxes, labels, scores = out[0]['boxes'], out[0]['labels'], out[0]['scores']
    #
    res = DeteRes(assign_img_path=assign_img_path)
    for index, each_box in enumerate(boxes):
        if float(scores[index]) > conf_th:
            x1, y1, x2, y2 = int(each_box[0]), int(each_box[1]), int(each_box[2]), int(each_box[3])
            conf, tag_index = float(scores[index]), str(labels[index].item())
            res.add_obj(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, tag=label_dict[int(tag_index)+1])

    # nms
    res.do_nms(0.1)
    # 保存画图和 xml
    img_name = os.path.split(assign_img_path)[1]
    save_img_path = os.path.join(assign_save_folder, img_name)
    save_xml_path = save_img_path[:-4] + '.xml'
    res.draw_dete_res(save_img_path)
    res.save_to_xml(save_xml_path)


if __name__ == "__main__":

    # todo 可以一次跑多个模型，并得到他们的效果对比

    # ----------------------------------------------------------------------------------------------------------------------
    args = args_parse()
    model_path = args['model']
    img_path = args['img_path']
    img_folder = args['img_folder']
    conf_th = float(args['conf_th'])
    save_folder = args['save_folder']
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
    # bg 是背景
    label_dict = ["fzc_yt", "fzc_sm", "fzc_gt", "fzc_other", "zd_yt", 'zd_sm', "zd_gt", "zd_other", "qx_yt", "qx_sm", "qx_gt", "other"]
    # ----------------------------------------------------------------------------------------------------------------------

    model = torch.load(model_path)
    model.cuda()
    model.eval()

    if os.path.isdir(img_folder):
        img_path_list = FileOperationUtil.re_all_file(img_folder, lambda x:str(x).endswith(('.jpg', '.JPG', '.png')))
        img_count = len(img_path_list)
        for img_index, each_img in enumerate(img_path_list):
            print_str = "{0}/{1} : {2}".format(img_index, img_count, each_img)
            print(print_str)
            dete_one_img(each_img, save_folder)
    else:
        if os.path.isdir(img_path):
            print(img_path)
            dete_one_img(img_path, save_folder)
        else:
            print("* 未发现需要检测的图片")
            print("img folder : {0}".format(img_folder))
            print("img path : {0}".format(img_path))



























