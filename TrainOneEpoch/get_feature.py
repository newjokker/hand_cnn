# -*- coding: utf-8  -*-
# -*- author: jokker -*-



import torchvision
from torchvision.models.detection import FasterRCNN
from PIL import Image
import torchvision.transforms as transforms



"""
* 输入 backbone 的 images 的 type 为 <class 'torchvision.models.detection.image_list.ImageList'>

"""



img_path = r'C:\Users\14271\Desktop\del\crop\test.jpg'
img = Image.open(img_path)
img = img.resize((800, 800))
transform = transforms.Compose([transforms.ToTensor()])
img = transform(img)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

images = [img]
images, targets = model.transform(images, targets=None)

print(type(images))

features = model.backbone(images.tensors, targets=None)

for f in features.values():
    print(f.size())
