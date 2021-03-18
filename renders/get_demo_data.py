from glob import glob
import numpy as np
import os 
import sys

WORK_DIR = '/home/admin_mcn/taint/code/rnd-semantic-segmentation'
DATA_DIR = "/home/admin_mcn/POLYP_DATA/BLI"
sys.path.append(WORK_DIR)

from core.utils.utility import dump_text

n = 20
imgfile = os.path.join(WORK_DIR, 'renders/demo_image.txt')
gtfile = os.path.join(WORK_DIR, 'renders/demo_groundtruth.txt')

image_paths = [img_path for img_path in glob(os.path.join(DATA_DIR, 'test') + '/*.JPG')]

random_picked = np.random.choice(len(image_paths), size=n, replace=False)
image_paths = [image_paths[index] for index in random_picked]
# label_paths = [os.path.join(DATA_DIR, 'masks', os.path.basename(img_path)) for img_path in image_paths]

dump_text(imgfile, image_paths)
# dump_text(gtfile, label_paths)