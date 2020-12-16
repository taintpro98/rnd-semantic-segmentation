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
        self.image_paths = []
        # for img_name in image_names:
        #     if self.check_gt(img_name):
        #         self.image_names.append(img_name)
        #     else:
        #         print("[WARNING] Can't found label for {}, skipped".format(img_name))

        kfolds = glob(data_root + "/*/")
        if mode == "train":
            for kfold_path in kfolds:
                if str(cross_val) not in os.path.basename(kfold_path):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]
        else:
            for kfold_path in kfolds:
                if str(cross_val) in os.path.basename(kfold_path):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]

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

    # def check_gt(self, img_name):
    #     path = os.path.join(self.data_root, 'masks', img_name)
    #     is_file = os.path.isfile(path) 
    #     return is_file

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        img_name = os.path.basename(self.image_names[index])
        img_dir = os.path.dirname(os.path.dirname(self.image_paths[index]))

        datafile = {
            'img': img_name,
            'label': os.path.join(img_dir, 'masks', img_name)
            'name': img_name[index][:-4]
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
