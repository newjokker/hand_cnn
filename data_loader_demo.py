# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# todo 掌握使用 pytorch 载入 voc ， coco 的分类数据和检测数据

# todo [×] 检测数据，voc 格式
# todo [×] 检测数据，coco 格式
# todo [×] 分类数据，voc 格式
# todo [×] 分类数据，coco 格式



def get_dataset(name, image_set, transform):
    paths = {
        "coco": ('/public/yzy/coco/2017/', get_coco, 91),
        "coco_kp": ('/datasets01/COCO/022719/', get_coco_kp, 2)}

    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes




dataset, num_classes = get_dataset(args.dataset, "train", get_transform(is_train=True))