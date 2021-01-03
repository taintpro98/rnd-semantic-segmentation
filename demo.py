import argparse
import os
import cv2
# import json
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
import torchvision.transforms as transforms

# from skimage.color import label2rgb
from PIL import Image
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from core.configs import cfg
from core.datasets.build import build_dataset, transform
from core.models.build import build_model, build_feature_extractor, build_classifier
# from core.solver import adjust_learning_rate
from core.utils.utility import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete, inference, strip_prefix_if_present, load_json, load_text
# from core.utils.logger import setup_logger
# from core.utils.metric_logger import MetricLogger
from core.models.classifiers.pranet.PraNet_Res2Net import PraNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = "configs/pranet_src_polyp.yaml"
cfg.merge_from_file(config_file)
# cfg.merge_from_list(None)
cfg.freeze()

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
    for idx, i in enumerate(images):
        if idx==1:
            res.append(np.array(i.convert('RGB')) * 255)
        else:
            res.append(np.array(i.convert('RGB')))
    res = np.stack(res, axis=0)
    res = res.transpose(0, 3, 1, 2)
    res = torch.from_numpy(res)
    return res
    
def dump_pr_curve(pred, label, id2name, writer):
    """
        :pred: numpy array (H*W, num_class)
        :label: numpy array (H * W, )
    """
    for clss in range(pred.shape[1]):
        labels = (label == clss) * 1
        predictions = pred[:, clss]
        writer.add_pr_curve(id2name[str(clss)], labels, predictions, clss)

def build_transform(name, cfg):
    w, h = cfg.INPUT.INPUT_SIZE_TEST
    if name == "aspp":
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    elif name == "pranet":
        def trans(image, label):
            img_transform = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            gt_transform = transforms.ToTensor()
            image = img_transform(image)
            return image, label
    return trans

def build_model(name, resume):
    if name == "aspp":
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
    elif name == "pranet":
        model = PraNet()
        model.load_state_dict(torch.load(resume, map_location=device))
        model.to(device)
        model.eval()
        return model

# def convert_pred2rgb(pred):
#     output = pred.max(1)[1]
#     output = output.numpy()
#     output = output.transpose(1,2,0)
#     label = output.squeeze(2)
#     return label2rgb(label)

def get_output(name, resume, image, label):
    """

        :return: numpy array H x W x C
    """
    image = image.to(device)
    if name == "aspp":
        feature_extractor, classifier = build_model(name, resume)
        output = inference(feature_extractor, classifier, image, label, flip=False) # tensor (B=1) x C x H x W    
        # pred = output.max(1)[1]
        output = output.cpu().numpy()
        output = output.transpose(0,2,3,1) 
        output = output.squeeze(0) # H * W * C numpy array with scores
        return output
    elif name == "pranet":
        gt = np.asarray(label, np.float32)
        gt /= (gt.max() + 1e-8)
        model = build_model(name, resume)
        
        res5, res4, res3, res2 = model(image)
        output = res2
        output = F.upsample(output, size=gt.shape, mode='bilinear', align_corners=False)
        output = output.sigmoid().data.cpu().numpy().squeeze()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        output = np.stack([1-output, output], axis=2) # create a H x W x 2 numpy array with 0 background and 1 polyp
        return output

def get_pred(output):
    """
        :output: numpy array H x W x C
        :return: numpy array H x W with classses
    """
    pred = output.argmax(2) # H * W numpy array with classes
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default='configs/demo_config.json', help='path to config')
    args = parser.parse_args()
    config = load_json(args.config_path)

    img_paths = load_text(config["sample"]["img_path"])
    lab_paths = load_text(config["sample"]["lab_path"])
    samples = list()

    transform = build_transform(config["name"], cfg)
    big_preds = [None] * len(config["weights"])
    big_label = None
    for img_path, lab_path in zip(img_paths, lab_paths):
        res = list()

        image = Image.open(img_path).convert('RGB')
        label = np.array(Image.open(lab_path), dtype=np.float32)
        if transform is not None:
            image, label = transform(image, label)
        image = image.unsqueeze(0)

        img = Image.open(img_path).convert('RGB')
        lab = Image.open(lab_path)
        res.append(img)
        res.append(lab.convert('RGB'))

        h, w = np.array(lab).shape
        tmp = np.array(lab).reshape(h*w)
        if big_label is None:
            big_label = tmp
        else:
            big_label = np.concatenate((big_label, tmp), axis=0)

        for idx, (k, resume) in enumerate(config["weights"].items()):
            output = get_output(config["name"], resume, image, label)
            pred = get_pred(output)
            result = get_color_pallete(pred, config["pallete"])

            h, w, c = output.shape
            tmp = output.reshape(h*w, c)
            if big_preds[idx] is None:
                big_preds[idx] = tmp
            else:
                big_preds[idx] = np.concatenate((big_preds[idx], tmp), axis=0)
            res.append(result)
        
        samples.append(res)
    
    if config["tensorboard"]:
        for idx, lab_path in enumerate(lab_paths):
            name = os.path.basename(lab_path)[:-4]
            label = np.array(Image.open(lab_path))
            label_copy = np.ones(label.shape, dtype=np.int32) * 255
            for k, v in config["id_to_trainid"].items():
                label_copy[label == int(k)] = v
            # label = Image.fromarray(label_copy)

            samples[idx] = convert_pilimgs2tensor(samples[idx])
            img_grid = torchvision.utils.make_grid(samples[idx])
            summary_writer = SummaryWriter(os.path.join(config["root"], config["name"], 'summary'))
            summary_writer.add_image(str(idx) + '.'+ name, img_grid)   
        summary_writer.close()

        for key, big_pred in zip(config["weights"].keys(), big_preds):
            writer = SummaryWriter(os.path.join(config["root"], config["name"], key))
            dump_pr_curve(big_pred, big_label, config["trainid2name"], writer)
            writer.close()

    else:
        for k, resume in config["weights"].items():
            pred = get_pred(resume)
            res = get_color_pallete(pred, config["pallete"])
            samples.append(res)
        
        n_rows = 1
        n_cols = len(samples)
        _, axs = plt.subplots(n_rows, n_cols, figsize=(30, 30))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            ax.imshow(img)
        plt.show()