import torch
import torch.nn as nn
import albumentations as al
import cv2

from torchvision.io import read_image
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms

from core.datasets import transform

def cv2_resize(image, label, size=(512, 512)):
    image = cv2.resize(image, dsize=size)
    if label is None:
        return image, None
    label = cv2.resize(label, dsize=size)
    return image, label

class Augmenter:
    def __init__(self, cfg, mode="train", is_source=True):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.is_source = is_source

    def build_transform(self):
        if self.cfg.AUG.NAME == "attn":
            return self.attn_trans()
        elif self.cfg.AUG.NAME == "pra":
            return self.pra_trans()
        elif self.cfg.AUG.NAME == "aspp":
            return self.aspp_trans()
        raise AttributeError("No Augmenter was required !")

    def attn_trans(self):
        def f(image, label):
            return image, label
        if self.mode == "train":
            # if not self.is_source:
            #     return f
            def F(image, label):
                trans = al.Compose([
                    al.MotionBlur(p=self.cfg.AUG.BLUR_PROB),
                    al.Rotate(p=self.cfg.AUG.ROTATE_PROB),
                    al.ColorJitter(p=self.cfg.AUG.JITTER_PROB),
                    al.Flip(p=self.cfg.AUG.FLIP_PROB)
                ], p=self.cfg.AUG.PROB)
                result = trans(image=image, mask=label)
                image, label = result["image"], result["mask"]
                return image, label
            return F
        elif self.mode == "test":
            return f

    def pra_trans(self):
        def F(image, label):
            w, h = self.cfg.INPUT.INPUT_SIZE_TEST
            if self.mode == "train":
                train_transform = Compose([
                    transforms.RandomRotate90(),
                    transforms.Flip(),
                    transforms.HueSaturationValue(),
                    transforms.RandomBrightnessContrast(),
                    transforms.Transpose(),
                    OneOf([
                        transforms.RandomCrop(220,220, p=0.5),
                        transforms.CenterCrop(220,220, p=0.5)
                    ], p=0.5),
                    transforms.Resize(self.trainsize, self.trainsize),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                augmented = train_transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']
            else:
                test_transform = Compose([
                    transforms.Transpose(),
                    transforms.Resize(self.trainsize, self.trainsize),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                augmented = test_transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']
            return image, label
        return F

    def aspp_trans(self):
        if self.mode=="train":
            w, h = self.cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if self.is_source else self.cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
            trans_list = [
                transform.ToTensor(),
                transform.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD, to_bgr255=self.cfg.INPUT.TO_BGR255)
            ]
            if self.cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
                trans_list = [transform.RandomHorizontalFlip(p=self.cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN),] + trans_list
            if self.cfg.INPUT.INPUT_SCALES_TRAIN[0]==self.cfg.INPUT.INPUT_SCALES_TRAIN[1] and self.cfg.INPUT.INPUT_SCALES_TRAIN[0]==1:
                trans_list = [transform.Resize((h, w)),] + trans_list
            else:
                trans_list = [
                    transform.RandomScale(scale=self.cfg.INPUT.INPUT_SCALES_TRAIN),
                    transform.RandomCrop(size=(h, w), pad_if_needed=True),
                ] + trans_list
            if self.is_source:
                trans_list = [
                    transform.ColorJitter(
                        brightness=self.cfg.INPUT.BRIGHTNESS,
                        contrast=self.cfg.INPUT.CONTRAST,
                        saturation=self.cfg.INPUT.SATURATION,
                        hue=self.cfg.INPUT.HUE,
                    ),
                ] + trans_list
            trans = transform.Compose(trans_list)
        else:
            w, h = self.cfg.INPUT.INPUT_SIZE_TEST
            trans = transform.Compose([
                transform.Resize((h, w), resize_label=False),
                transform.ToTensor(),
                transform.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD, to_bgr255=self.cfg.INPUT.TO_BGR255)
            ])
        return trans

    # def forward(self, image_t, mask_t):
    #     image_n = image_t.numpy().transpose(1, 2, 0)
    #     mask_n = mask_t.numpy().transpose(1, 2, 0)

    #     result = self.transforms(image=image_n, mask=mask_n)

    #     image_n, mask_n = result["image"], result["mask"]
    #     image_t = torch.from_numpy(image_n).permute(2, 0, 1)
    #     mask_t = torch.from_numpy(mask_n).permute(2, 0, 1)

    #     return image_t, mask_t