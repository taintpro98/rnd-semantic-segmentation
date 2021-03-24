from glob import glob
import numpy as np
import os 
import sys

WORK_DIR = '/home/admin_mcn/taint/code/rnd-semantic-segmentation'
DATA_DIR = "/home/admin_mcn/taint/dataset/cityscapes"
sys.path.append(WORK_DIR)

from core.utils.utility import dump_text

n = 100
imgfile = os.path.join(WORK_DIR, 'renders/demo_image.txt')
gtfile = os.path.join(WORK_DIR, 'renders/demo_groundtruth.txt')

# image_paths = [img_path for img_path in glob(os.path.join(DATA_DIR, 'images') + '/*.jpeg')]
image_paths = list()
img_dirs = glob(os.path.join(DATA_DIR, "leftImg8bit/%s" % "test") + "/*/")
for img_dir in img_dirs:
    image_paths += [img_path for img_path in glob(img_dir + '/*.png')]

random_picked = np.random.choice(len(image_paths), size=n, replace=False)
image_paths = [image_paths[index] for index in random_picked]

# label_paths = [os.path.join(DATA_DIR, 'masks', os.path.basename(img_path)[:-5] + '.png') for img_path in image_paths]
label_paths = list()
for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img_dir = os.path.basename(os.path.dirname(img_path))
    label_paths.append(os.path.join(DATA_DIR, "gtFine", "test", img_dir, img_name.split("_leftImg8bit")[0] + "_gtFine_labelIds.png"))

dump_text(imgfile, image_paths)
dump_text(gtfile, label_paths)