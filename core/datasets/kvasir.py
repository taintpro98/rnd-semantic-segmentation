import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import pickle
from glob import glob

class KvasirDataSet(Dataset):
    def __init__(self, data_root, num_classes=2, mode="train", cross_val=0, transform=None, ignore_label=255, debug=False):
        super(KvasirDataSet, self).__init__()

        self.data_root = data_root
        image_names = [file for file in os.listdir(os.path.join(self.data_root, 'images')) if file.endswith('.png') or file.endswith('.jpg')]
        
        self.image_names = []
        for img_name in image_names:
            if self.check_gt(img_name):
                self.image_names.append(img_name)
            else:
                print("[WARNING] Can't found label for {}, skipped".format(img_name))

        kfolds = glob(data_root + "/*/")

        self.trainid2name = {
            0: "background",
            1: "polyp"
        }
        self.id_to_trainid = {
            0: 0, 1: 1
        }
        self.debug = debug
        self.ignore_label = ignore_label
        self.transform = transform

    def check_gt(self, img_name):
        path = os.path.join(self.data_root, 'masks', img_name)
        is_file = os.path.isfile(path) 
        return is_file

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafile = {
            'img': os.path.join(self.data_root, 'images', self.image_names[index]),
            'label': os.path.join(self.data_root, 'masks', self.image_names[index]),
            'name': self.image_names[index][:-4]
        }

        image = Image.open(datafile["img"]).convert('RGB')
        label = np.array(Image.open(datafile["label"]),dtype=np.float32)
        name = datafile["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, name
