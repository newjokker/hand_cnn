import os
import math
import time
import random
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

from config import DefaultConfig
from transforms import test_transforms
from my_dataset import VOCDataset
from eval import *


model_path = ""
test_dir = ""


def detect(opt, config):
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    transforms = test_transforms()
    img_list = [img for img in os.listdir(test_dir) if img.endswith(".jpg")]
    print("total_images : {}".format(len(img_list)))

    model.eval()
    print("=== eval ===")
    pred_boxes = []
    pred_classes = []
    pred_scores = []

    for i, img_name in enumerate(img_list):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
        img = transforms(image=img)['image'] 
        inputs = [img]

        with torch.no_grad():
            pred = model(inputs)
        tmp = pred[i]["boxes"].cpu()
        pred_boxes.append(tmp)
        tmp = pred[i]["labels"].cpu()
        pred_classes.append(tmp)
        tmp = pred[i]["scores"].cpu()
        pred_scores.append(tmp)

        del inputs
        del pred
        del tmp
        print("{} / {}".format(i+1, len(img_list), end='\r'))

    

def fix_seed(config):
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="gpu during training")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_config = DefaultConfig
    fix_seed(train_config)

    detect(args, train_config)
