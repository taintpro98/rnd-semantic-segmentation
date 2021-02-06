import os
import numpy as np
from glob import glob
import random

import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class cityscapesDataSet(data.Dataset):
    def __init__(self, data_root, num_classes=19, mode="train", transform=None, ignore_label=255, debug=False):
        self.mode = mode
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.image_paths = list()
        img_dirs = glob(os.path.join(self.data_root, "leftImg8bit/%s" % self.mode) + "/*/")
        
        for img_dir in img_dirs:
            self.image_paths += [img_path for img_path in glob(img_dir + '/*.png')]

        # for fname in content:
        #     name = fname.strip()
        #     self.data_list.append(
        #         {
        #             "img": os.path.join(
        #                 self.data_root, "leftImg8bit/%s/%s" % (self.mode, name)
        #             ),
        #             "label": os.path.join(
        #                 self.data_root, "gtFine/%s/%s" % (self.split, name.split("_leftImg8bit")[0] + "_gtFine_labelIds.png"),
        #             ),
        #             "name": name,
        #         }
        #     )

        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
        if self.NUM_CLASS==16:# SYNTHIA 
            self.id_to_trainid = {
                7: 0,
                8: 1,
                11: 2,
                12: 3,
                13: 4,
                17: 5,
                19: 6,
                20: 7,
                21: 8,
                23: 9,
                24: 10,
                25: 11,
                26: 12,
                28: 13,
                32: 14,
                33: 15,
            }
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        img_name = os.path.basename(self.image_paths[index])
        img_dir = os.path.basename(os.path.dirname(self.image_paths[index]))
        
        datafile = {
            'img': self.image_paths[index],
            'label': os.path.join(self.data_root, "gtFine", self.mode, img_dir, img_name.split("_leftImg8bit")[0] + "_gtFine_labelIds.png"),
            'name': img_name[:-4]
        }

        image = Image.open(datafile["img"]).convert('RGB')
        label = np.array(Image.open(datafile["label"]),dtype=np.float32)
        name = datafile["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # for k in self.trainid2name.keys():
        #     label_copy[label == k] = k
        label = Image.fromarray(label_copy)
        
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, name