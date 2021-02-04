# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
from torch import nn
from torchvision import models


vgg = models.vgg16()

print(vgg.features)
print(vgg.classifier)






