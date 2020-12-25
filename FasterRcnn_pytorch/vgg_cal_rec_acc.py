# -*- coding: utf-8  -*-
# -*- author: jokker -*-


from JoTools.utils.FileOperationUtil import FileOperationUtil
import prettytable
import os
from JoTools.utils.NumberUtil import NumberUtil
from JoTools.utils.JsonUtil import JsonUtil
import uuid
import torch
import cv2
import numpy as np
import shutil


def cal_acc_classify(standard_img_dir, customized_img_dir):
    """"对比两个分类结果文件夹，分类就是将原图进行了重新的排列"""

    # 拿到标签
    return_res = []
    standard_dict = {}
    stand_label_count = {}
    res_dict = {}
    for each_img_path in FileOperationUtil.re_all_file(standard_img_dir, lambda x:str(x).endswith(('.jpg', '.JPG', '.png'))):
        # 拿到第一级别文件夹名，作为 label
        img_label = each_img_path[len(standard_img_dir):].strip(os.sep).split(os.sep)[0]
        img_name = os.path.split(each_img_path)[1]
        standard_dict[img_name] = img_label
        if img_label in stand_label_count:
            stand_label_count[img_label] += 1
        else:
            stand_label_count[img_label] = 1
    #
    for each_img_path in FileOperationUtil.re_all_file(customized_img_dir, lambda x:str(x).endswith(('.jpg', '.JPG', '.png'))):
        # 拿到第一级别文件夹名，作为 label
        img_label = each_img_path[len(customized_img_dir):].strip(os.sep).split(os.sep)[0]
        img_name = os.path.split(each_img_path)[1]
        #
        standard_img_label = standard_dict[img_name]
        #
        if standard_img_label == img_label:
            correct_str = "correct_{0}".format(standard_img_label)
            if correct_str in res_dict:
                res_dict[correct_str].append(each_img_path)
            else:
                res_dict[correct_str] = [each_img_path]
        else:
            mistake_str = "mistake_{0}_{1}".format(standard_img_label, img_label)
            if mistake_str in res_dict:
                res_dict[mistake_str].append(each_img_path)
            else:
                res_dict[mistake_str] = [each_img_path]

    stand_label_list = list(stand_label_count.keys())
    tb = prettytable.PrettyTable()
    tb.field_names = ["  ", "class", "num", "per"]

    # 计算每一个类型的召回率
    for each in stand_label_list:
        correct_str = "correct_{0}".format(each)
        if correct_str in res_dict:
            # print(correct_str, len(res_dict[correct_str]), NumberUtil.format_float(len(res_dict[correct_str])/stand_label_count[each], 2))
            rec = NumberUtil.format_float(len(res_dict[correct_str])/stand_label_count[each], 2)
            one_row = ['rec', each, "{0} | {1}".format(len(res_dict[correct_str]), stand_label_count[each]), rec]
            tb.add_row(one_row)
            return_res.append(one_row)

    # 计算每一个类型的准确率
    for i in stand_label_list:
        correct_str = "correct_{0}".format(i)
        # 去掉没检测出来的类型
        if correct_str not in res_dict:
            continue
        #
        correct_num = len(res_dict[correct_str])
        all_num = correct_num
        for j in stand_label_list:
            mistake_str = "mistake_{0}_{1}".format(j, i)
            if mistake_str in res_dict:
                all_num += len(res_dict[mistake_str])
        # print("rec {0} : {1}".format(i, NumberUtil.format_float(correct_num/all_num), 2))
        acc = NumberUtil.format_float(correct_num/all_num, 2)
        one_row = ['acc', i, "{0} | {1}".format(correct_num, all_num), acc]
        tb.add_row(one_row)
        return_res.append(one_row)

    mistake_tb = prettytable.PrettyTable()
    mistake_tb.field_names = ["correct", "mistake", "num"]

    for i in stand_label_list:
        for j in stand_label_list:
            mistake_str = "mistake_{0}_{1}".format(i, j)
            if mistake_str in res_dict:
                # print(mistake_str, len(res_dict[mistake_str]))
                mistake_tb.add_row([i, j, len(res_dict[mistake_str])])

    print(tb)
    print(mistake_tb)
    return return_res


def classify_one_img(assign_img_path, assign_save_folder, label_list):
    """检测一张图片"""
    # src_img = cv2.imread(assign_img_path)
    src_img = cv2.imdecode(np.fromfile(assign_img_path, dtype=np.uint8), 1)
    src_img = cv2.resize(src_img, (224, 224))
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()

    # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cpu()
    img_tensor = torch.unsqueeze(img_tensor, 0)
    out = model(img_tensor)
    pred = out.data.max(1, keepdim=True)[1]
    pre = pred.data.item()

    save_label_folder = os.path.join(assign_save_folder, str(label_list[pre]))
    if not os.path.exists(save_label_folder): os.makedirs(save_label_folder)
    img_name = os.path.split(assign_img_path)[1]
    save_path = os.path.join(save_label_folder, img_name)
    shutil.copy(assign_img_path, save_path)


def test_one_model(model_path, standard_dir, tmp_dir, label_list, save_dir):
    """检测一个模型，生成模型性能 txt """

    model = torch.load(model_path)
    model.to(device)
    model.eval()
    tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1()))
    img_path_list = FileOperationUtil.re_all_file(standard_dir, lambda x: str(x).endswith(('.jpg', '.JPG', '.png')))
    img_count = len(img_path_list)
    for img_index, each_img in enumerate(img_path_list):
        print_str = "{0}/{1} : {2}".format(img_index, img_count, each_img)
        print(print_str)
        classify_one_img(each_img, tmp_dir, label_list)
    # 计算模型性能
    each_res_dict = cal_acc_classify(standard_dir, tmp_dir)
    # 模型数据保存戴指定位置
    model_name = os.path.split(model_path)[1].strip('.pth')
    save_path = os.path.join(save_dir, model_name + '.json')
    JsonUtil.save_data_to_json_file(each_res_dict, save_path)
    # 删除临时文件夹



if __name__ == "__main__":

    # fixme 批量计算模型性能

    # ------------------------------------------------------------------------------------------------------------------
    label_list = ["yt", "sm", "gt", "other", "zd_yt", "fzc_broken"]
    standard_dir = r"/home/ldq/002_test_data/crop_add_broken"
    tmp_dir = r"/home/ldq/004_tmp_dir"
    save_dir = r"/home/ldq/003_test_res"
    model_dir = r"/home/ldq/test_fzc_classify/models/crop_add_broken"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
    # ------------------------------------------------------------------------------------------------------------------

    model_list = FileOperationUtil.re_all_file(model_dir, lambda x:str(x).endswith('.pth'))

    for each_model_path in model_list:
        model = torch.load(each_model_path)
        model.to(device)
        model.eval()
        tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1()))
        img_path_list = FileOperationUtil.re_all_file(standard_dir, lambda x: str(x).endswith(('.jpg', '.JPG', '.png')))
        img_count = len(img_path_list)
        for img_index, each_img in enumerate(img_path_list):
            print_str = "{0}/{1} : {2}".format(img_index, img_count, each_img)
            print(print_str)
            classify_one_img(each_img, tmp_dir, label_list)
        # 计算模型性能
        each_res_dict = cal_acc_classify(standard_dir, tmp_dir)
        # 模型数据保存戴指定位置
        model_name = os.path.split(each_model_path)[1].strip('.pth')
        save_path = os.path.join(save_dir, model_name + '.json')
        JsonUtil.save_data_to_json_file(each_res_dict, save_path)
        # 删除临时文件夹



