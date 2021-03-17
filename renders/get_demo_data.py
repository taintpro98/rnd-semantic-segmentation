from glob import glob
import numpy as np
import os 

import sys
sys.path.append('/Users/macbook/Documents/AI/topics/DomainAdaptation/code/rnd-semantic-segmentation')
from core.utils.utility import dump_text

DATA_DIR = "/Users/macbook/Documents/AI/projects/VINIF/dataset/Kvasir_fold_new/fold_0/"
n = 20
imgfile = '/Users/macbook/Documents/AI/topics/DomainAdaptation/code/rnd-semantic-segmentation/configs/demo_image.txt'
gtfile = '/Users/macbook/Documents/AI/topics/DomainAdaptation/code/rnd-semantic-segmentation/configs/demo_groundtruth.txt'

image_paths = [img_path for img_path in glob(os.path.join(DATA_DIR, 'images') + '/*.png')]

random_picked = np.random.choice(len(image_paths), size=n, replace=False)
image_paths = [image_paths[index] for index in random_picked]
label_paths = [os.path.join(DATA_DIR, 'masks', os.path.basename(img_path)) for img_path in image_paths]

dump_text(imgfile, image_paths)
dump_text(gtfile, label_paths)