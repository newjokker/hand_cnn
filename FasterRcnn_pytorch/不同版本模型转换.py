# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torch
import torchvision

# todo 直接将新版本保存数据，不保存模型即可
# torch.save(model.state_dict(), "my_model.pth")  # 只保存模型的参数


def state_dict_to_model(model, state_dict_path, model_path):
    """将模型文件转为模型加模型文件"""
    model_state_dict = torch.load(state_dict_path)
    model.load_state_dict(model_state_dict)
    torch.save(model, model_path)


if __name__ == "__main__":

    assigin_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=17, pretrained_backbone=True)
    model_dict_path = r""
    save_model_path = r""

    state_dict_to_model(assigin_model, model_state_dict, save_model_path)


