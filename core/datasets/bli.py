import os
import numpy as np
from PIL import Image
from glob import glob

from torch.utils.data import Dataset
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms
from skimage.io import imread

class BLIDataset(Dataset):
    def __init__(self, cfg, data_root, trainsize=352, num_classes=2, mode="train", transform=None, ignore_label=255, debug=False):
        super(BLIDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.trainsize = trainsize
        self.data_root = data_root
        self.image_paths = list()
        self.trainid2name = {
            0: "background",
            1: "polyp"
        }
        self.id_to_trainid = {
            0: 0, 1: 1
        }
        self.image_paths += [img_path for img_path in glob(os.path.join(data_root, 'train') + '/*.*') if img_path.endswith("JPG") or img_path.endswith("jpg") or img_path.endswith("png")]

        self.debug = debug
        self.ignore_label = ignore_label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_name = os.path.basename(self.image_paths[index])
        datafile = {
            'img': self.image_paths[index],
            'label': None,
            'name': img_name[:-4]
        }
        # image = Image.open(datafile['img']).convert('RGB')
        image = imread(datafile["img"])
        # mask = imread(datafile["label"])
        image, _ = self._transform(image=image, label=image)

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        name = datafile['name']

        return image, 0, name

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
