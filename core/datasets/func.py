import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as ttf

from core.utils.utility import Threshold

def attn_collate_fn(batch):
    images, masks, names = zip(*batch)

    images = torch.stack([torch.from_numpy(image) for image in images], 0) # tensor B x H x W x 3
    images = images.permute(0, 3, 1, 2) # tensor B x 3 x H x W
    images = images.float() / 255.

    masks = torch.stack([
        torch.from_numpy(mask[:,:,np.newaxis]) for mask in masks
    ], 0) # tensor B x H x W x 1
    masks = masks.long()
    masks = masks.permute(0, 3, 1, 2) # tensor B x 1 x H x W
    # masks = ttf.Compose([
    #     ttf.Grayscale(num_output_channels=1),
    #     Threshold(threshold=128),
    # ])(masks)
    return images, masks, names

def collate_fn(batch):
    """
    :param batch: 
    :return:
        Tensor (B x C x H x W)
    """
    images, masks, names = zip(*batch)

    images = torch.stack([torch.from_numpy(image) for image in images], 0) # tensor B x H x W x 3
    images = images.permute(0, 3, 1, 2) # tensor B x 3 x H x W
    images = images.float() / 255.

    if np.isscalar(masks[0]):
        return images, 0, names

    masks = torch.stack([
        torch.from_numpy(mask[:,:,np.newaxis]) for mask in masks
    ], 0) # tensor B x H x W x 1
    masks = masks.long()
    masks = masks.permute(0, 3, 1, 2) # tensor B x 1 x H x W
    return images, masks, names