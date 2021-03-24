import os
import numpy as np
from PIL import Image
from glob import glob
import cv2

from torch.utils.data import Dataset
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms
from skimage.io import imread
from core.components.augment import cv2_resize

class BLIDataset(Dataset):
    def __init__(self, cfg, data_root, mode="train", transform=None, debug=False):
        super(BLIDataset, self).__init__()
        self.cfg = cfg
        self.data_root = data_root 
        self.mode = mode
        self.transform = transform
        self.debug = debug

        self.image_paths = list()
        self.id_to_trainid = {
            0: 0, 1: 1
        }
        self.image_paths += [img_path for img_path in glob(os.path.join(data_root, 'images') + '/*.*') if img_path.endswith("JPG") or img_path.endswith("jpg") or img_path.endswith("png") or img_path.endswith("jpeg")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.debug:
            index = 0

        img_name = os.path.basename(self.image_paths[index])
        img_dir = os.path.dirname(os.path.dirname(self.image_paths[index]))
        datafile = {
            'img': self.image_paths[index],
            'label': os.path.join(img_dir, 'masks', img_name[:-5] + '.png'),
            'name': img_name[:-5]
        }
        image = imread(datafile["img"]) # numpy array RGB (not BGR like OpenCV)
        # mask = imread(datafile["label"])
        mask = cv2.imread(datafile["label"])
        mask = mask / 255
        mask = mask.astype(np.uint8)
        mask = mask[:, :, 0]
        name = datafile["name"]

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        if self.mode == "train":
            image, mask = cv2_resize(image, mask, self.cfg.INPUT.TARGET_INPUT_SIZE_TRAIN)
        else:
            image, mask = cv2_resize(image, mask, self.cfg.INPUT.INPUT_SIZE_TEST)
        return image, mask, name

    def _transform(self, image, label):
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
        return image, None