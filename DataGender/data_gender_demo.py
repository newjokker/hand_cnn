# -*- coding: utf-8  -*-
# -*- author: jokker -*-


from dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
import os
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------------------------------------------------------
"""
* training_generator 是个字典，有三个键，annot，img，scale
    * annot, 的shape为 [batch_size, 标记的个数, 5], 其中的 5 代表五个参数，前四个是坐标信息，最后一个是 tag 信息
    * img, 的 shape 为 [batch_size, 3, img_size, img_size] 
    * scale, 为一个 list，里面可能放的是缩放系数，元素个数等于 batch_size，我看到的值为 0.768 ，我看了代码这边代表的是图片的缩放值大小，因为放入检测的图可能与需要的尺寸不相符合，需要进行缩放

"""


class Params:
    """yml 文件解析"""

    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)



if __name__ == "__main__":


    # ----------------------------------------------------------------------------------------------------------------------

    data_path = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/datasets"
    project_name = r"fzc_broken_class"
    yaml_path = r"/home/ldq/Yet-Another-EfficientDet-Pytorch/projects/fzc_broken_class.yml"
    batch_size = 1
    num_workers = 12  # 并行的个数
    compound_coef = 7
    # ----------------------------------------------------------------------------------------------------------------------

    # 解析 yml 文件
    params = Params(yaml_path)

    training_params = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True, 'collate_fn': collater, 'num_workers': num_workers}
    val_params = {'batch_size': batch_size, 'shuffle': False, 'drop_last': True, 'collate_fn': collater, 'num_workers': num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    training_set = CocoDataset(root_dir=os.path.join(data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std), Augmenter(), Resizer(input_sizes[compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std), Resizer(input_sizes[compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)


    print("OK")


































