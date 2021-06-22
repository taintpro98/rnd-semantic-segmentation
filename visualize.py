from core.utils.utility import plot_confusion_matrix, LineChartPlotter, load_json, moving_average, plot_images
import numpy as np

from core.datasets import transform
import cv2
from PIL import Image

input_file = "results/src_gald/gald_chart_params.json"
loss_filepath = "/Users/macbook/Desktop/loss.png"
lr_filepath = "/Users/macbook/Desktop/lr.png"

loss_plotter = LineChartPlotter("GALD Source Training", "iteration", "loss", loss_filepath)
# lr_plotter = LineChartPlotter("GALD Source Training", "iteration", "learning rate", lr_filepath)

data = load_json(input_file)
for key, value in data.items():
    if "loss" in key:
        chart = {
            "x": range(1, len(moving_average(value))+1),
            "y": moving_average(value),
            "label": key
        }
        loss_plotter.add_chart(chart)
        
    else:
        # chart = {
        #     "x": range(1, len(value)+1),
        #     "y": value,
        #     "label": key
        # }
        # lr_plotter.add_chart(chart)
        continue
loss_plotter.display()
# lr_plotter.display()

# json_input = "/Users/macbook/Downloads/aspp_confusion_matrix.json"
# data = load_json(json_input)
# cmt = np.array(data['cmt'])
# classes = data['classes']
# plot_confusion_matrix(cmt, classes)

# PIXEL_MEAN = [0.485, 0.456, 0.406]
# PIXEL_STD = [0.229, 0.224, 0.225]
# TO_BGR255 = False

# inputfile = "/Users/macbook/Documents/AI/topics/DomainAdaptation/dataset/GTA5/images/00192.png"
# labelfile = "/Users/macbook/Documents/AI/topics/DomainAdaptation/dataset/GTA5/labels/02191.png"

# image = Image.open(inputfile).convert('RGB')
# label = np.array(Image.open(labelfile), dtype=np.float32)

# images = [np.array(image)]
# labels = [label]
# titles = ["Raw"]

# trans_list = [
#     transform.ToTensor(),
#     transform.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD, to_bgr255=TO_BGR255)
# ]
# trans = transform.Compose(trans_list)
# image_fn, label_fn = trans(image, label)
# images.append(image_fn.numpy().transpose(1,2,0))
# titles.append("Normalize")

# trans_list = [
#     transform.ToTensor(),
#     transform.RandomHorizontalFlip(p=1)
# ]
# trans = transform.Compose(trans_list)
# image_fn, label_fn = trans(image, label)
# images.append(image_fn.numpy().transpose(1,2,0))
# titles.append("Horizontal Flip")

# image_fn = cv2.resize(np.array(image), dsize=(512, 512))
# images.append(image_fn)
# titles.append("Resize")

# trans_list = [
#     # transform.RandomScale(scale=(0.5, 1.5)),
#     transform.RandomCrop(size=(512, 1024), pad_if_needed=True),
#     transform.ToTensor()
# ]
# trans = transform.Compose(trans_list)
# print(label)
# image_fn, label_fn = trans(image, label)
# images.append(image_fn.numpy().transpose(1,2,0))
# titles.append("RandomScale")

# trans_list = [
#     transform.ColorJitter(
#         brightness=0.5,
#         contrast=0.5,
#         saturation=0.5,
#         hue=0.2,
#     ),
#     transform.ToTensor(),
# ]
# trans = transform.Compose(trans_list)
# image_fn, label_fn = trans(image, label)
# images.append(image_fn.numpy().transpose(1,2,0))
# titles.append("ColorJitter")


# plot_images(images, titles, 'preprocessing.png')