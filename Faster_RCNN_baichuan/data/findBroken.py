import os
import numpy as np
from PIL import Image


img_dir = "VOC2007/JPEGImages"
img_list = os.listdir(img_dir)
print("total: ", len(img_list))
for i, img in enumerate(img_list):
    print(i+1) 
    absolute_path = os.path.join(img_dir, img)

    try:
        img = Image.open(absolute_path)
    except IOError:
        print(absolute_path)
    try:
        img= np.array(img, dtype=np.float32)
    except :
        print('corrupt img',absolute_path)
        break
