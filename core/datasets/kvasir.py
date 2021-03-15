import os
import numpy as np
from PIL import Image
from glob import glob

from torch.utils.data import Dataset

class KvasirDataSet(Dataset):
    def __init__(self, data_root, num_classes=2, mode="train", cross_val=0, transform=None, ignore_label=255, debug=False):
        super(KvasirDataSet, self).__init__()
        self.data_root = data_root        
        self.image_paths = []

        kfolds = glob(data_root + "/*/")
        if mode == "train":
            for kfold_path in kfolds:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]
        else:
            for kfold_path in kfolds:
                if str(cross_val) in os.path.basename(kfold_path[:-1]):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        img_name = os.path.basename(self.image_paths[index])
        img_dir = os.path.dirname(os.path.dirname(self.image_paths[index]))

        datafile = {
            'img': self.image_paths[index],
            'label': os.path.join(img_dir, 'masks', img_name),
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

# class KvasirFoldDataset(Dataset):
#     def __init__(self, root_dir: str,
#         shape=(512, 512),
#         image_dir="images",
#         mask_dir="masks",
#         return_paths=False,
#         augmenter: Augmenter = Augmenter()
#     ):
#         self.root_dir = root_dir
#         self.shape = shape
#         self.augmenter = augmenter
#         self.return_paths = return_paths

#         self.__image_dir = image_dir
#         self.__mask_dir = mask_dir
#         self.__scan_files()

#     def __scan_files(self):
#         self.__pairs = []

#         for image_name in os.listdir(self.image_dir):
#             ext = image_name.split('.')[-1]
#             image_path = os.path.join(self.image_dir, image_name)
#             mask_path = os.path.join(self.mask_dir, image_name)

#             if ext not in ('jpg', 'png'):
#                 logger.warning(f'Skipping file {image_path}')
#                 continue

#             if not os.path.exists(mask_path):
#                 logger.warning(f'No mask found for {image_path}')
#                 continue

#             self.__pairs.append({
#                 'image': image_path,
#                 'mask': mask_path
#             })

#     def __len__(self) -> int:
#         return len(self.__pairs)

#     def __getitem__(self, index: int):
#         pair = self.__pairs[index]
#         image_path, mask_path = pair['image'], pair['mask']
#         resizer = ttf.Resize(self.shape)

#         # Read images
#         image_t = read_image(image_path)
#         mask_np = self.read_binary_mask(mask_path)
#         if mask_np is None:
#             raise ValueError(f"Cannot read file at {mask_path}")
#         mask_t = torch.from_numpy(mask_np)

#         # Augment first
#         image_t, mask_t = self.augmenter(image_t, mask_t)

#         # Resize
#         image_t = resizer(image_t)
#         mask_t = resizer(mask_t)

#         if not self.return_paths:
#             return image_t, mask_t
#         else:
#             return image_t, mask_t, image_path, mask_path

#     @property
#     def image_dir(self) -> str:
#         return os.path.join(self.root_dir, self.__image_dir)

#     @property
#     def mask_dir(self) -> str:
#         return os.path.join(self.root_dir, self.__mask_dir)

#     @property
#     def meta_path(self) -> str:
#         return os.path.join(self.root_dir, "meta.yml")

#     @staticmethod
#     def read_binary_mask(path):
#         mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         mask = mask * 255
#         mask = np.stack([mask, mask, mask])
#         return mask


# class KvasirMultiFoldDataset(ConcatDataset):
#     def __init__(self, root_dirs: List[str], **kwargs):
#         super().__init__([
#             KvasirFoldDataset(root_dir, **kwargs)
#             for root_dir in root_dirs
#         ])
