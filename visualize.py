from core.utils.utility import plot_confusion_matrix, LineChartPlotter, load_json, moving_average
import numpy as np

from core.datasets import transform
import cv2

# input_file = "results/src_r101_try/aspp_chart_params.json"
# loss_filepath = "/Users/macbook/Desktop/loss.png"
# lr_filepath = "/Users/macbook/Desktop/lr.png"

# loss_plotter = LineChartPlotter("ASPP Source Training", "iteration", "loss", loss_filepath)
# lr_plotter = LineChartPlotter("ASPP Source Training", "iteration", "learning rate", lr_filepath)

# data = load_json(input_file)
# for key, value in data.items():
#     if "loss" in key:
#         chart = {
#             "x": range(1, len(moving_average(value))+1),
#             "y": moving_average(value),
#             "label": key
#         }
#         loss_plotter.add_chart(chart)
#     else:
#         chart = {
#             "x": range(1, len(value)+1),
#             "y": value,
#             "label": key
#         }
#         lr_plotter.add_chart(chart)
# loss_plotter.display()
# lr_plotter.display()

trainid2name = {
        "0": "road",
        "1": "sidewalk",
        "2": "building",
        "3": "wall",
        "4": "fence",
        "5": "pole",
        "6": "light",
        "7": "sign",
        "8": "vegetation",
        "9": "terrain",
        "10": "sky",
        "11": "person",
        "12": "rider",
        "13": "car",
        "14": "truck",
        "15": "bus",
        "16": "train",
        "17": "motorcycle",
        "18": "bicycle"
    }

json_input = "/Users/macbook/Downloads/aspp_confusion_matrix.json"
data = load_json(json_input)
cmt = np.array(data['cmt'])
classes = data['classes']
plot_confusion_matrix(cmt, classes)

# PIXEL_MEAN = [0.485, 0.456, 0.406]
# PIXEL_STD = [0.229, 0.224, 0.225]
# TO_BGR255 = False

# inputfile = "/Users/macbook/Documents/AI/topics/DomainAdaptation/dataset/GTA5/images/02191.png"
# preprocess = cv2.imread(inputfile)

# trans_list = [
#     transform.ToTensor(),
#     transform.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD, to_bgr255=TO_BGR255)
# ]
# trans = transform.Compose(trans_list)
