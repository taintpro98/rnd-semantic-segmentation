import torch
import torch.nn as nn

from torchvision import transforms as ttf
from core.utils.utility import Threshold

def attn_collate_fn(batch):
    image_t = torch.stack([
        b[0] for b in batch
    ])
    mask_t = torch.stack([
        b[1] for b in batch
    ], dim=0)  # B x 3 x H x W

    image_t = image_t.float() / 255.

    mask_t = ttf.Compose([
        ttf.Grayscale(num_output_channels=1),
        Threshold(threshold=128),
    ])(mask_t)
    mask_t = mask_t.long()

    return image_t, mask_t

def pra_collate_fn(batch):
    """
    :param batch: 
    :return:
        Tensor (B x C x H x W)
    """
    images, masks, names = zip(*batch)

    images = torch.stack(images, 0) # tensor B x H x W x 3
    images = images.permute(0, 3, 1, 2) # tensor B x 3 x H x W
    images = images.float() / 255.

    masks = torch.stack([
        mask[:,:,np.newaxis] for mask in masks
    ], 0) # tensor B x H x W x 1
        
    masks = masks.float()
    masks = masks.permute(0, 3, 1, 2) # tensor B x 1 x H x W

    return images, masks, names