import os
import os.path as osp
import numpy as np
import random
from glob import glob
import matplotlib.pyplot as plt
import collections

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import pickle

class CarlaFoldDataSet(Dataset):
    def __init__(self, cfg, data_root, mode="train", cross_val=0, transform=None, debug=False, ignore_label=255):
        super(CarlaFoldDataSet, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug
        self.num_class = cfg.MODEL.NUM_CLASSES
        # self.image_paths = [img_path for img_path in glob(os.path.join(data_root, 'images') + '/*.png')]
        self.image_paths = list()

        kfolds = glob(data_root + "/*/")
        if mode == "train":
            for kfold_path in kfolds:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]
        else:
            for kfold_path in kfolds:
                if str(cross_val) in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]

        self.id_to_trainid = {0: 0, 11: 1, 13: 2, 24: 4, 17: 5,
                              7: 6, 7: 7, 8: 8, 21: 9, 26: 10, 12: 11, 20: 12}

        self.trainid2name = {
            0:"none",
            1:"building",
            2:"fence",
            3:"other",
            4:"pedestrian", #person
            5:"pole",
            6:"roadline", #road
            7:"road",
            8:"sidewalk",
            9:"vegetation",
            10:"vehicle", #car
            11:"wall",
            12:"trafficsign"
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        img_name = os.path.basename(self.image_paths[index])
        img_dir = os.path.dirname(os.path.dirname(self.image_paths[index]))

        datafile = {
            'img': self.image_paths[index],
            'label': os.path.join(img_dir, 'labels', img_name),
            'name': img_name[:-4]
        }

        image = Image.open(datafile["img"]).convert('RGB')
        label = np.array(Image.open(datafile["label"]), dtype=np.float32)
        name = datafile["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, name
