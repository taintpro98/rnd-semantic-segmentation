import os
import numpy as np
from PIL import Image
from glob import glob

from torch.utils.data import Dataset

class BLIDataset(Dataset):
    def __init__(self, data_root, num_classes=2, mode="train", transform=None, ignore_label=255, debug=False):
        super(BLIDataset, self).__init__()
        self.data_root = data_root
        self.image_paths = list()
        self.trainid2name = {
            0: "background",
            1: "polyp"
        }
        self.id_to_trainid = {
            0: 0, 1: 1
        }
        self.image_paths += [img_path for img_path in glob(os.path.join(data_root, 'images') + '/*.*') if img_path.endswith("JPG") or img_path.endswith("jpg") or img_path.endswith("png")]

        self.debug = debug
        self.ignore_label = ignore_label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_name = os.path.basename(self.image_paths[index])
        datafile = {
            'img': self.image_paths[index],
            'name': img_name[:-4]
        }
        image = Image.open(datafile['img']).convert('RGB')
        name = datafile['name']
        return image, name
