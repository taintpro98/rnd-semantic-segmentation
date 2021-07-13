import argparse
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from core.configs import cfg
from core.models.build import build_feature_extractor, build_classifier
from core.utils.utility import strip_prefix_if_present, load_json, inference, get_color_palette
from core.models.classifiers.gcpacc.gcpa_cc2 import GCPADecoder, GCPAEncoder
from core.datasets import transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_transform(name, cfg, img_path, lab_path):
    w, h = cfg.INPUT.INPUT_SIZE_TEST
    trans = transform.Compose([
        transform.Resize((h, w), resize_label=False),
        transform.ToTensor(),
        transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
    ])
    image = Image.open(img_path).convert('RGB')
    height, width, _ = np.array(image).shape
    label = np.array(Image.open(lab_path), dtype=np.float32) if config["labeled"] else np.zeros((height, width))
    image, label = trans(image, label)
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
        checkpoint = torch.load(resume, map_location=device)
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        return feature_extractor, classifier
    elif name.startswith('gald'):
        encoder = GCPAEncoder()
        decoder = GCPADecoder()
        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        checkpoint = torch.load(resume, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        return encoder, decoder

def get_output(cfg, name, resume, image, label):
    """
        :return: numpy array H x W x C
    """
    image = image.to(device)
    if name.startswith("aspp"):
        feature_extractor, classifier = build_model(cfg, name, resume)
        output = inference(feature_extractor, classifier, image, label, flip=False) # tensor (B=1) x C x H x W    
        # output = multi_scale_inference(feature_extractor, classifier, image, label, flip=False) # tensor (B=1) x C x H x W           
        # pred = output.max(1)[1]
        output = output.cpu().numpy()
        output = output.transpose(0,2,3,1) 
        output = output.squeeze(0) # H * W * C numpy array with scores
    elif name.startswith('gald'):
        encoder, decoder = build_model(cfg, name, resume)
        h, w = label.size()[-2:]
        hardnetout = encoder(image)
        res5, res4, res3, res2 = decoder(image, hardnetout)
        res = res2
        res = F.upsample(
            res, size=(h, w), mode="bilinear", align_corners=False
        ) # tensor [(B=1) x C x H x W]
        output = F.softmax(res, dim=1)
        output = output.cpu().detach().numpy()
        output = output.transpose(0,2,3,1)
        output = output.squeeze(0)
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

    img_path = "../datasets/cityscapes/leftImg8bit/val/munster/munster_000030_000019_leftImg8bit.png"
    lab_path = "../datasets/cityscapes/gtFine/val/munster/munster_000030_000019_gtFine_color.png"
    resume = "../weights/Gald-8.pth"
    image, label = build_transform(config["name"], cfg, img_path, lab_path)
    height, width = label.shape[:2]

    img = Image.open(img_path).convert('RGB') 
    lab = Image.open(lab_path) if config["labeled"] else Image.fromarray(np.zeros((height, width)))

    if config["labeled"] == "id":
        gt = np.array(lab, dtype=np.float32)
        label_copy = cfg.MODEL.NUM_CLASSES * np.ones(gt.shape[:2], dtype=np.float32)
        for k, v in config["id_to_trainid"].items():
            label_copy[gt == int(k)] = v
        # gt = Image.fromarray(label_copy)
        gt = label_copy
        palette = config["palette"] + [0, 0, 0]
        result = get_color_palette(gt, palette)
    output = get_output(cfg, config["name"], resume, image, label)
    pred = get_pred(output)
    if config["labeled"] == "id":
        pred[gt == cfg.MODEL.NUM_CLASSES] = cfg.MODEL.NUM_CLASSES
        result = get_color_palette(pred, config["palette"] + [0, 0, 0])
    else:
        result = get_color_palette(pred, config["palette"])
    result.save(config["name"] + ".png")