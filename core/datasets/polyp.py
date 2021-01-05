import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from glob import glob

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, data_root, mode="train", cross_val=0, trainsize=352, transform=None):
        super(PolypDataset, self).__init__()
        self.trainsize = trainsize
        self.data_root = data_root        
        self.image_paths = []
        self.transform = transform
        self.mode = mode

        kfolds = glob(data_root + "/*/")
        if mode == "train":
            for kfold_path in kfolds:
                if str(cross_val) not in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]
        else:
            for kfold_path in kfolds:
                if str(cross_val) in os.path.basename(kfold_path[:-1]):
                    self.image_paths += [img_path for img_path in glob(os.path.join(kfold_path, 'images') + '/*.png')]

    def __len__(self):
        return len(self.image_paths)    

    def __getitem__(self, index):
        img_name = os.path.basename(self.image_paths[index])
        img_dir = os.path.dirname(os.path.dirname(self.image_paths[index]))

        datafile = {
            'img': self.image_paths[index],
            'label': os.path.join(img_dir, 'masks', img_name),
            'name': img_name[:-4]
        }
        image = self.rgb_loader(datafile["img"])
        gt = self.binary_loader(datafile["label"])

        image, gt = self._transform(image, gt)
        name = datafile["name"]
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def _transform(self, image, label):
        if self.mode == "train":
            img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
            gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
            image = img_transform(image)
            gt = gt_transform(gt)
        else:
            img_transform = transforms.Compose([
                transforms.Resize((w, h)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            image = img_transform(image)
            gt = np.array(gt.convert('L'))
            gt = torch.from_numpy(gt).unsqueeze(0)
        return image, label