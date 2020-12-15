import argparse
import os
import json
# import datetime
# import logging
# import time
# import math
import numpy as np
from collections import OrderedDict
import re
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import torch
import torchvision
import torch.nn.functional as F
from skimage.color import label2rgb
from PIL import Image
from tensorboardX import SummaryWriter

from core.configs import cfg
from core.datasets import build_dataset, transform
from core.models import build_model, build_feature_extractor, build_classifier
# from core.solver import adjust_learning_rate
from core.utils.utility import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete, inference, strip_prefix_if_present
# from core.utils.logger import setup_logger
# from core.utils.metric_logger import MetricLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = "configs/deeplabv2_r101_tgt_self_distill.yaml"
cfg.merge_from_file(config_file)
# cfg.merge_from_list(None)
cfg.freeze()

id_to_trainid = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}
cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
gta5_trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
synthia_trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }
trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }

# COLORS = ('white','red', 'blue', 'yellow', 'magenta', 
#             'green', 'indigo', 'darkorange', 'cyan', 'pink', 
#             'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
#             'purple', 'darkviolet')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def convert_pilimgs2tensor(images):
    res = []
    for i in images:
        res.append(np.array(i.convert('RGB')))
    res = np.stack(res, axis=0)
    res = res.transpose(0, 3, 1, 2)
    res = torch.from_numpy(res)
    return res
    
def dump_pr_curve(pred, label, id2name, writer):
    """
        :pred: numpy array H x W x num_class
        :label: numpy array H x W x 1
    """
    for clss in range(pred.shape[2]):
        labels = (label == clss) * 1
        predictions = pred[:, :, clss]
        writer.add_pr_curve(id2name[clss], labels, predictions, clss)

def build_transform(cfg):
    w, h = cfg.INPUT.INPUT_SIZE_TEST
    trans = transform.Compose([
        transform.Resize((h, w), resize_label=False),
        transform.ToTensor(),
        transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
    ])
    return trans

def build_model(resume):
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    feature_extractor.eval()
    classifier.eval()
    checkpoint = torch.load(resume, map_location=torch.device('cpu'))
    feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
    feature_extractor.load_state_dict(feature_extractor_weights)
    classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
    classifier.load_state_dict(classifier_weights)
    return feature_extractor, classifier

# def convert_pred2rgb(pred):
#     output = pred.max(1)[1]
#     output = output.numpy()
#     output = output.transpose(1,2,0)
#     label = output.squeeze(2)
#     return label2rgb(label)

def get_pred(resume):
    feature_extractor, classifier = build_model(resume)
    pred = inference(feature_extractor, classifier, image, label, flip=False)
    return pred

def add_curve(writer, pred, label):
    pred = pred.numpy()
    pred = pred.transpose(0,2,3,1)
    pred = pred.squeeze(0)
    dump_pr_curve(pred, label, trainid2name, writer)

src_resume = "/Users/macbook/Documents/AI/DomainAdaptation/weights/fada/25-10-2020/src.pth"
adv_resume = "/Users/macbook/Documents/AI/DomainAdaptation/weights/fada/25-10-2020/adv.pth"
sd_resume = "/Users/macbook/Documents/AI/DomainAdaptation/weights/fada/25-10-2020/sd.pth"

# img_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/GTA5/images/00756.png"
# lab_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/GTA5/labels/00756.png"

img_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/leftImg8bit_trainvaltest/leftImg8bit/train/monchengladbach/monchengladbach_000000_010733_leftImg8bit.png"
lab_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/gtFine_trainvaltest/gtFine/train/monchengladbach/monchengladbach_000000_010733_gtFine_color.png"
bin_lab_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/gtFine_trainvaltest/gtFine/train/monchengladbach/monchengladbach_000000_010733_gtFine_labelIds.png"

# img_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/leftImg8bit_trainvaltest/leftImg8bit/val/munster/munster_000031_000019_leftImg8bit.png"
# lab_path = "/Users/macbook/Documents/AI/DomainAdaptation/dataset/gtFine_trainvaltest/gtFine/val/munster/munster_000031_000019_gtFine_color.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default='demo_config.json', help='path to config')
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    img_path = config["sample"]["img_path"]
    lab_path = config["sample"]["lab_path"]
    samples = list()

    image = Image.open(img_path).convert('RGB')
    label = np.array(Image.open(lab_path),dtype=np.float32)
    
    transform = build_transform(cfg)
    if transform is not None:
        image, label = transform(image, label)
    image = image.unsqueeze(0)

    img = Image.open(img_path).convert('RGB')
    lab = Image.open(lab_path).convert('RGB')
    samples.append(img)
    samples.append(lab)
    
    if config["tensorboard"]:
        label = np.array(Image.open(bin_lab_path))
        label_copy = np.ones(label.shape, dtype=np.int32) * 255
        for k, v in config["id_to_trainid"].items():
            label_copy[label == int(k)] = v
        # label = Image.fromarray(label_copy)

        for k, resume in config["weights"].items():
            writer = SummaryWriter(os.path.join(config["root"], k))
            pred = get_pred(resume)
            res = get_color_pallete(pred)
            add_curve(writer, pred, label_copy)
            writer.close()
            samples.append(res)

        samples = convert_pilimgs2tensor(samples)
        img_grid = torchvision.utils.make_grid(samples)
        writer = SummaryWriter(os.path.join(config["root"], 'summary'))
        writer.add_image('5_fada_images', img_grid)                
        writer.close()

    else:
        for k, resume in config["weights"].items():
            pred = get_pred(resume)
            res = get_color_pallete(pred)
            samples.append(res)
        
        n_rows = 1
        n_cols = len(samples)
        _, axs = plt.subplots(n_rows, n_cols, figsize=(30, 30))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img)
        plt.show()
    
