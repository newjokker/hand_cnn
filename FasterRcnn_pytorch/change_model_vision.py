# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torch
import torchvision

def save_state_dict(model_path, save_path="my_model.pth"):
    """从模型中提取数据并保存"""
    model = torch.load(model_path)
    torch.save(model.state_dict(), save_path)  # 只保存模型的参数


def state_dict_to_model(model, state_dict_path, model_path):
    """将模型文件转为模型加模型文件"""
    model_state_dict = torch.load(state_dict_path)
    model.load_state_dict(model_state_dict)
    torch.save(model, model_path)


if __name__ == "__main__":

    assigin_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=13, pretrained_backbone=True)
    model_dict_path = r""
    save_model_path = r""

    state_dict_to_model(assigin_model, model_dict_path, save_model_path)


