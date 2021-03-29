import argparse
import os
import cv2
import numpy as np
from collections import OrderedDict
import re
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

import torch
import torchvision
import torch.nn.functional as F
# import torchvision.transforms as transforms

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

from core.datasets import transform

# from skimage.color import label2rgb
from PIL import Image
from skimage.io import imread

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from core.configs import cfg
from core.datasets.build import build_dataset
from core.models.build import build_feature_extractor, build_classifier
from core.utils.utility import mkdir, get_color_palette, inference, strip_prefix_if_present, load_json, load_text
from core.models.classifiers.pranet.PraNet_Res2Net import PraNet
from core.models.classifiers.attn.eff import Encoder, Decoder, AttnEfficientNetUnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COLORS = ('white','red', 'blue', 'yellow', 'magenta', 
#             'green', 'indigo', 'darkorange', 'cyan', 'pink', 
#             'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
#             'purple', 'darkviolet')

# def convert_pred2rgb(pred):
#     output = pred.max(1)[1]
#     output = output.numpy()
#     output = output.transpose(1,2,0)
#     label = output.squeeze(2)
#     return label2rgb(label)

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
            res.append(np.array(i.convert('RGB')))
            # res.append(np.array(i.convert('RGB')) * 255)
        else:
            res.append(np.array(i.convert('RGB')))
    res = np.stack(res, axis=0)
    res = res.transpose(0, 3, 1, 2)
    res = torch.from_numpy(res)
    return res
        
def convert_pilimgs2npgrid(images, pad=255):
    raw = np.array(images[0])
    h, _, _ = raw.shape
    padd = np.ones((h, 50, 3)) * pad
    res = raw
    for i in images[1:]:
        res = np.concatenate((res, padd), axis=1)
        res = np.concatenate((res, np.array(i.convert('RGB'))), axis=1)
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

def build_transform(name, cfg, img_path, lab_path):
    w, h = cfg.INPUT.INPUT_SIZE_TEST
    if name.startswith("aspp"):
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
        image = Image.open(img_path).convert('RGB')
        height, width, _ = np.array(image).shape
        label = np.array(Image.open(lab_path), dtype=np.float32) if config["labeled"] else np.zeros((height, width))
        
        image, label = trans(image, label)
    elif name.startswith("pranet"):
        # def trans(image, label):
        #     img_transform = transforms.Compose([
        #     transforms.Resize((w, h)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #     gt_transform = transforms.ToTensor()
        #     image = img_transform(image)
        #     return image, label
        def trans(image, label):
            test_transform = Compose([
                transforms.Transpose(),
                transforms.Resize(h, w),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            augmented = test_transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            return image, label
        image = imread(img_path)
        label = imread(lab_path) if config["labeled"] else np.zeros((image.shape[0], image.shape[1]))
        image, label = trans(image, label)
    elif name.startswith("attn"):
        image = imread(img_path)
        label = imread(lab_path) if config["labeled"] else np.zeros((image.shape[0], image.shape[1]))
        image = cv2.resize(image, dsize=(w, h))  
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.

    image = image.unsqueeze(0)
    return image, label

def build_model(cfg, name, resume):
    if name.startswith('aspp'):
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
    elif name.startswith('pranet'):
        model = PraNet()
        model.load_state_dict(torch.load(resume, map_location=device))
        model.to(device)
        model.eval()
        return model
    elif name.startswith('attn'):
        encoder = Encoder(backbone_name="efficientnet-b2")
        decoder = Decoder(backbone_name="efficientnet-b2")
        encoder.to(device)
        decoder.to(device)
        checkpoint = torch.load(resume, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder.eval()
        decoder.eval()
        return encoder, decoder

def get_output(cfg, name, resume, image, label):
    """

        :return: numpy array H x W x C
    """
    image = image.to(device)
    if name.startswith("aspp"):
        feature_extractor, classifier = build_model(cfg, name, resume)
        output = inference(feature_extractor, classifier, image, label, flip=False) # tensor (B=1) x C x H x W    
        # pred = output.max(1)[1]
        output = output.cpu().numpy()
        output = output.transpose(0,2,3,1) 
        output = output.squeeze(0) # H * W * C numpy array with scores
    elif name.startswith("pranet"):
        gt = np.asarray(label, np.float32)
        gt /= (gt.max() + 1e-8)
        model = build_model(cfg, name, resume)

        res5, res4, res3, res2 = model(image)
        output = res2
        output = F.upsample(output, size=gt.shape, mode='bilinear', align_corners=False)
        output = output.sigmoid().data.cpu().numpy().squeeze()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        output = np.stack([1-output, output], axis=2) # create a H x W x 2 numpy array with 0 background and 1 polyp
    elif name.startswith("attn"):
        encoder, decoder = build_model(cfg, name, resume)
        endpoints = encoder(image)
        outputs = decoder(endpoints) 
        output = outputs[0]
        output = F.interpolate(output, size=label.shape, mode='bilinear', align_corners=True) # resize output to original size
        output = torch.sigmoid(output)  # tensor (B=1) x C x H x W
        output = output.cpu().detach().numpy() # detach to unable grad ???
        output = output.transpose(0,2,3,1) # numpy array (B=1) x H x W x C
        output = output.squeeze(0) # H * W * C numpy array with scores
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
    parser.add_argument(
        "-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "-c", 
        "--config_path", 
        default="renders/cityscapes.json", 
        metavar="FILE", 
        help='path to config', 
        type=str
    )

    args = parser.parse_args()
    config = load_json(args.config_path)

    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(None)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    print("Loaded demo parameters from file {}".format(args.config_path))

    img_paths = load_text(config["sample"]["img_path"])
    lab_paths = load_text(config["sample"]["lab_path"])
    samples = list()

    big_preds = [None] * len(config["weights"])
    big_label = None

    if config["tensorboard"]:
        summary_writer = SummaryWriter(os.path.join(config["root"], config["name"], 'summary'))

    for index, (img_path, lab_path) in enumerate(zip(img_paths, lab_paths)):
        res = list()
        image, label = build_transform(config["name"], cfg, img_path, lab_path)
        height, width = label.shape[:2]

        img = Image.open(img_path).convert('RGB') 
        lab = Image.open(lab_path) if config["labeled"] else Image.fromarray(np.zeros((height, width)))

        res.append(img)
        if config["labeled"] == "id":
            gt = np.array(lab)
            label_copy = cfg.MODEL.NUM_CLASSES * np.ones(gt.shape[:2], dtype=np.float32)
            for k, v in config["id_to_trainid"].items():
                label_copy[gt == k] = v
            gt = Image.fromarray(label_copy)
            palette = config["palette"] + [0, 0, 0]
            result = get_color_palette(gt, palette)
            res.append(result)
        else:
            res.append(lab.convert('RGB'))

        if config["tensorboard"]:
            h, w = np.array(lab).shape[:2]
            tmp = np.array(lab).reshape(h*w)
            if big_label is None:
                big_label = tmp
            else:
                big_label = np.concatenate((big_label, tmp), axis=0)

        for idx, (k, resume) in enumerate(config["weights"].items()):
            output = get_output(cfg, config["name"], resume, image, label)
            pred = get_pred(output)
            if config["labeled"] == "id":
                pred[gt == cfg.MODEL.NUM_CLASSES] = cfg.MODEL.NUM_CLASSES
                result = get_color_palette(pred, config["palette"] + [0, 0, 0])
            else:
                result = get_color_palette(pred, config["palette"])


            h, w, c = output.shape
            tmp = output.reshape(h*w, c)
            if big_preds[idx] is None:
                big_preds[idx] = tmp
            else:
                big_preds[idx] = np.concatenate((big_preds[idx], tmp), axis=0)
            res.append(result)
        
        # samples.append(res)
        print("{}. Dumped images of {} into a grid".format(index+1, os.path.basename(lab_path)))
        if config["tensorboard"]:
            name = os.path.basename(lab_path)[:-4]
            # label = np.array(Image.open(lab_path))
            # label_copy = np.ones(label.shape, dtype=np.int32) * 255
            # for k, v in config["id_to_trainid"].items():
            #     label_copy[label == int(k)] = v
            # label = Image.fromarray(label_copy)

            res = convert_pilimgs2tensor(res)
            img_grid = torchvision.utils.make_grid(res)
            summary_writer.add_image(str(index) + '.'+ name, img_grid)   

        else:
            filename = os.path.basename(lab_path)
            res = convert_pilimgs2npgrid(res, 255)
            # cv2.imwrite(os.path.join(config["dir"], filename), res)
            Image.fromarray(res.astype(np.uint8)).save(os.path.join(config["dir"], filename))
            print("{}, saved {} in {}".format(index+1, filename, config["dir"]))

    if config["tensorboard"]:
        summary_writer.close()
        for key, big_pred in zip(config["weights"].keys(), big_preds):
            writer = SummaryWriter(os.path.join(config["root"], config["name"], key))
            dump_pr_curve(big_pred, big_label, config["trainid2name"], writer)
            writer.close()
    
    # if config["tensorboard"]:
    #     for idx, lab_path in enumerate(lab_paths):
    #         name = os.path.basename(lab_path)[:-4]
    #         # label = np.array(Image.open(lab_path))
    #         # label_copy = np.ones(label.shape, dtype=np.int32) * 255
    #         # for k, v in config["id_to_trainid"].items():
    #         #     label_copy[label == int(k)] = v
    #         # label = Image.fromarray(label_copy)

    #         samples[idx] = convert_pilimgs2tensor(samples[idx])
    #         img_grid = torchvision.utils.make_grid(samples[idx])
            
    #         summary_writer = SummaryWriter(os.path.join(config["root"], config["name"], 'summary'))
    #         summary_writer.add_image(str(idx) + '.'+ name, img_grid)   
    #     summary_writer.close()

    #     for key, big_pred in zip(config["weights"].keys(), big_preds):
    #         writer = SummaryWriter(os.path.join(config["root"], config["name"], key))
    #         dump_pr_curve(big_pred, big_label, config["trainid2name"], writer)
    #         writer.close()

    # else:
    #     for idx, lab_path in enumerate(lab_paths):
    #         filename = os.path.basename(lab_path)
    #         samples[idx] = convert_pilimgs2npgrid(samples[idx], 255)
    #         cv2.imwrite(os.path.join(config["dir"], filename), samples[idx])
    #         print("{}, saved {} in {}".format(idx+1, filename, config["dir"]))