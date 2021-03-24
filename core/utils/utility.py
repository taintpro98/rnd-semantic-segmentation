import json
import numpy as np
from PIL import Image
import logging
import os 

from collections import defaultdict
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1-Score = 2 * Precision * Recall / (Precision + Recall)
        """
        self.intersection_sum = 0 # TP
        self.union_sum = 0 # FP + TP + FN
        self.target_sum = 0 # TP + FN (target)
        self.res_sum = 0 # FP + TP (output)
        self.count = 0 
        self.iou_sum = 0 # TP/(FP + TP + FN)
        self.f1_sum = 0 # 2*TP/(2*TP + FP + FN) = 2*inter/(output + target)

    def update(self, intersection, union, target, res):
        iou = intersection/(union + 1e-10)
        f1 = 2*intersection/(target + res + 1e-10)

        self.intersection_sum += intersection
        self.union_sum += union
        self.target_sum += target
        self.res_sum += res
        self.count += 1
        self.iou_sum += iou
        self.f1_sum += f1

    def summary(self, logger, num_classes=2):
        macro_f1 = self.f1_sum/float(self.count)
        macro_iou = self.iou_sum/float(self.count)

        micro_f1 = 2*self.intersection_sum/(self.target_sum + self.res_sum + 1e-10)
        micro_iou = self.intersection_sum/(self.union_sum + 1e-10)

        mIoU = np.mean(macro_iou)
        mF1 = np.mean(macro_f1)
        logger.info('Macro metric, val result: mIoU/mF1 {:.4f}/{:.4f}.'.format(mIoU, mF1))

        mIoU = np.mean(micro_iou)
        mF1 = np.mean(micro_f1)
        logger.info('Micro metric, val result: mIoU/mF1 {:.4f}/{:.4f}.'.format(mIoU, mF1))

        for i in range(num_classes):
            logger.info('Macro metric, class {} iou/f1 score: {:.4f}/{:.4f}.'.format(i, macro_iou[i], macro_f1[i]))
            logger.info('Micro metric, class {} iou/f1 score: {:.4f}/{:.4f}.'.format(i, micro_iou[i], micro_f1[i]))

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target, area_output


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda(), area_output.cuda()

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def inference(feature_extractor, classifier, image, label, flip=True):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        output = classifier(feature_extractor(image))
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    return output.unsqueeze(dim=0)

def get_color_palette(pred, palette):
    """
        :pred: H * W numpy array with 0 and 1
    """
    label = Image.fromarray(pred.astype('uint8')).convert('P')
    label.putpalette(palette)
    return label

def load_json(json_path):
    with open(json_path) as ref:
        data = json.load(ref)
    return data

def load_text(path):
    with open(path, "r") as ref:
        samples = [line.rstrip() for line in ref]
    return samples

def dump_text(path, data):
    with open(path, 'w') as ref:
        for line in data:
            ref.write("%s\n" % line)

def setup_logger(name, save_dir, distributed_rank):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, name + ".txt")),
            logging.StreamHandler()
        ]
    )    
    #Creating an object 
    logger=logging.getLogger(name) 
    return logger

class Threshold(nn.Module):
    """
    Thresholding transform for image
    """
    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def forward(self, input):
        mask = input >= self.threshold
        return torch.as_tensor(mask, dtype=torch.int)

def generate_scales(input, scales):
    """
    Generate scaled versions of a tensor

    :param input: Input tensor
    :type input: torch.Tensor
    :param scales: List of scales to generate
    :type scales: list
    :return: List of scaled tensors
    """
    height, width = input.shape[-2:]
    outputs = []

    for scale in scales:
        sw = int(width * scale)
        sh = int(height * scale)
        out = transforms.Resize((sh, sw))(input)
        outputs.append(out)

    return outputs

def probs_to_mask(probs, thres=0.5):
    """
    Convert a probability tensor into a class mask

    :param probs: Tensor of shape [B x C x H x W], values from 0 to 1
    :return: Tensor of shape [B x H x W], each pixel corresponds to a class
    """
    num_classes = probs.shape[1]

    if num_classes > 1:
        return torch.argmax(probs, dim=1)
    else:
        return (probs > thres).int()


def probs_to_onehot(probs, thres=0.5):
    """
    Converts a [B x C x H x W] probability map to a one-hot encoded matrix

    :param probs: The input mask
    :return: Output one-hot mask
    """
    num_classes = probs.shape[1]
    if num_classes > 1:
        indices = torch.argmax(probs, dim=1)
        return F.one_hot(indices, num_classes).permute(0, 3, 1, 2)
    else:
        return (probs > thres).int()

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val):
#         self.val = val
#         self.sum += val
#         self.count += 1
#         self.avg = self.sum / self.count