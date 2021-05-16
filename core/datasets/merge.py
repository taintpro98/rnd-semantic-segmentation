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

class MergeDataSet(Dataset):
    def __init__(self, cfg, data_root, mode="train", cross_val=0, transform=None, debug=False, ignore_label=255):
        super(MergeDataSet, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.sub_dir_1 = os.path.join(data_root, "gta5"),
        self.sub_dir_2 = os.path.join(data_root, "carla"),

        self.mode = mode
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug
        self.num_class = cfg.MODEL.NUM_CLASSES
        # self.image_paths = [img_path for img_path in glob(os.path.join(data_root, 'images') + '/*.png')]
        self.image_paths = list()

        kfolds_1 = glob(self.sub_dir_1 + "/*/") #gta
        kfolds_2 = glob(self.sub_dir_2 + "/*/") #carla

        if mode == "train":
            #img_path in gta5
            for kfold_path in kfolds_1:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]

            #img_path in carla
            for kfold_path in kfolds_2:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]
        else:
            #img_path in gta5
            for kfold_path in kfolds_1:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]

            #img_path in carla
            for kfold_path in kfolds_2:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]

        self.id_to_trainid_1 = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.id_to_trainid_2 = {0: 0, 11: 1, 13: 2, 24: 4, 17: 5,
                              7: 6, 7: 7, 8: 8, 21: 9, 26: 10, 12: 11, 20: 12}

        self.trainid2name_1 = {
            0:"road",
            1:"sidewalk",
            2:"building",
            3:"wall",
            4:"fence",
            5:"pole",
            6:"light",
            7:"sign",
            8:"vegetation",
            9:"terrain",
            10:"sky",
            11:"person",
            12:"rider",
            13:"car",
            14:"truck",
            15:"bus",
            16:"train",
            17:"motocycle",
            18:"bicycle"
        }

        self.trainid2name_2 = {
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

        if 'gta5' in datafile["img"]:
            id_to_trainid = self.id_to_trainid_1
        elif 'carla' in datafile["img"]:
            id_to_trainid = self.id_to_trainid_2

        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, name
