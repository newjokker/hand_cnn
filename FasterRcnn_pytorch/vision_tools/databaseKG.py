# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import os
import copy
import random
from JoTools.txkj.parseXml import ParseXml, parse_xml
from JoTools.utils.FileOperationUtil import FileOperationUtil
from PIL import Image



class DatabaseKG():
    """开口销使用的函数"""

    # todo 图像随机旋转，只旋转 90 的倍数，还是用黑框补全

    @staticmethod
    def is_in_range(range_0, range_1):
        """判断一个范围是不是包裹另外一个范围，(xmin, ymin, xmax, ymax)"""
        x_min_0, y_min_0, x_max_0, y_max_0 = range_0
        x_min_1, y_min_1, x_max_1, y_max_1 = range_1
        #
        if x_min_0 < x_min_1:
            return False
        elif x_max_0 > x_max_1:
            return False
        elif y_min_0 < y_min_1:
            return False
        elif y_max_0 > y_max_1:
            return False
        else:
            return True

    @staticmethod
    def merge_range_list(range_list):
        """进行区域合并得到大的区域"""
        x_min_list, y_min_list, x_max_list, y_max_list = [], [], [], []
        for each_range in range_list:
            x_min_list.append(each_range[0])
            y_min_list.append(each_range[1])
            x_max_list.append(each_range[2])
            y_max_list.append(each_range[3])
        return (min(x_min_list), min(y_min_list), max(x_max_list), max(y_max_list))

    @staticmethod
    def get_subset_from_pic(xml_path, img_path, save_dir, min_count=6, small_img_count=3):
        """从一个图片中拿到标签元素的子集"""
        xml_operate = ParseXml()
        xml_info = xml_operate.get_xml_info(xml_path)
        box_list = []
        for each in xml_info["object"]:
            each_range = (int(each["bndbox"]["xmin"]), int(each["bndbox"]["ymin"]), int(each["bndbox"]["xmax"]), int(each["bndbox"]["ymax"]))
            box_list.append(each_range)

        if len(box_list) < 1:
            return

        # fixme 都有全局的图，要是元素大于等于 6 还另外生成小图

        # 图片中的所有元素放到一个小截图中
        merge_range = DatabaseKG.merge_range_list(box_list)
        # 修改数据范围
        xml_info_tmp = copy.deepcopy(xml_info)
        for each_obj in xml_info_tmp["object"]:
            each_obj["bndbox"]["xmin"] = str(int(each_obj["bndbox"]["xmin"]) - merge_range[0] )
            each_obj["bndbox"]["ymin"] = str(int(each_obj["bndbox"]["ymin"]) - merge_range[1] )
            each_obj["bndbox"]["xmax"] = str(int(each_obj["bndbox"]["xmax"]) - merge_range[0] )
            each_obj["bndbox"]["ymax"] = str(int(each_obj["bndbox"]["ymax"]) - merge_range[1] )
        # 存储 xml 和 jpg
        xml_info_tmp["size"] = {'width': str(merge_range[2]-merge_range[0]), 'height': str(merge_range[3]-merge_range[1]), 'depth': '3'}
        each_xml_save_path = os.path.join(save_dir, os.path.split(xml_path)[1])
        xml_operate.save_to_xml(each_xml_save_path, xml_info_tmp)
        # 剪切图像
        each_jpg_save_path = os.path.join(save_dir, os.path.split(xml_path)[1][:-4] + ".jpg")
        #
        # img_path = xml_path[:-4] + ".jpg"
        img = Image.open(img_path)
        each_crop = img.crop(merge_range)
        each_crop.save(each_jpg_save_path)

        # 元素个数大于阈值，另外生成小图
        if len(box_list) > min_count:
            for i in range(small_img_count):
                # xml_info_tmp = xml_info.copy()
                xml_info_tmp = copy.deepcopy(xml_info)      # copy() 原来不是深拷贝啊，不是直接开辟空间存放值？
                # 打乱顺序
                random.shuffle(box_list)
                # 拿出其中的三个，得到外接矩形的外接矩形
                merge_range = DatabaseKG.merge_range_list(box_list[:3])
                # 遍历所有要素，找到在 merge_range 中的要素，
                obj_list = []
                for each_obj in xml_info_tmp["object"]:
                    each_range = (int(each_obj["bndbox"]["xmin"]), int(each_obj["bndbox"]["ymin"]), int(each_obj["bndbox"]["xmax"]), int(each_obj["bndbox"]["ymax"]))
                    # print(each_range, merge_range)
                    if DatabaseKG.is_in_range(each_range, merge_range):
                        # 裁剪后的图像范围要进行对应的平移
                        each_obj["bndbox"]["xmin"] = str(int(each_obj["bndbox"]["xmin"]) - merge_range[0])
                        each_obj["bndbox"]["ymin"] = str(int(each_obj["bndbox"]["ymin"]) - merge_range[1])
                        each_obj["bndbox"]["xmax"] = str(int(each_obj["bndbox"]["xmax"]) - merge_range[0])
                        each_obj["bndbox"]["ymax"] = str(int(each_obj["bndbox"]["ymax"]) - merge_range[1])
                        obj_list.append(each_obj)
                #
                xml_info_tmp["object"] = obj_list
                xml_info_tmp["size"] = {'width': str(merge_range[2]-merge_range[0]), 'height': str(merge_range[3]-merge_range[1]), 'depth': '3'}
                xml_name = "_{0}.xml".format(i)
                img_name = "_{0}.jpg".format(i)
                xml_info_tmp["filename"] = img_name                                                             # 修改文件名
                each_xml_save_path = os.path.join(save_dir, os.path.split(xml_path)[1][:-4] + xml_name)
                xml_operate.save_to_xml(each_xml_save_path, xml_info_tmp)
                # 剪切图像
                each_jpg_save_path = os.path.join(save_dir, os.path.split(xml_path)[1][:-4] + img_name)
                # img_path = xml_path[:-4] + ".jpg"
                img = Image.open(img_path)
                each_crop = img.crop(merge_range)
                each_crop.save(each_jpg_save_path)



if __name__ == "__main__":


    # --------------------------------------------------------------------------------------------------------
    xmlDir = r"C:\Users\14271\Desktop\kkx_xml\xml"
    imgDir = r"C:\Users\14271\Desktop\kkx_xml\img"
    saveDir = r"C:\Users\14271\Desktop\kkx_clc"

    for index, each_xml_path in enumerate(FileOperationUtil.re_all_file(xmlDir, lambda x:str(x).endswith((".xml")))):
        print(index, each_xml_path)

        each_img_path = os.path.join(imgDir, os.path.split(each_xml_path)[1][:-4] + '.jpg')
        DatabaseKG.get_subset_from_pic(each_xml_path, each_img_path, saveDir, min_count=4, small_img_count=20)
    # --------------------------------------------------------------------------------------------------------
