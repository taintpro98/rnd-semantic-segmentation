from core.utils.utility import plot_confusion_matrix, LineChartPlotter, load_json, moving_average
import numpy as np

input_file = "results/adv_gald/gald_fada_chart_params.json"
loss_filepath = "/Users/macbook/Desktop/loss.png"
lr_filepath = "/Users/macbook/Desktop/lr.png"

loss_plotter = LineChartPlotter("Gald Adversarial", "iteration", "loss", loss_filepath)
lr_plotter = LineChartPlotter("Gald Adversarial", "iteration", "learning rate", lr_filepath)

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
        chart = {
            "x": range(1, len(value)+1),
            "y": value,
            "label": key
        }
        lr_plotter.add_chart(chart)
loss_plotter.display()
lr_plotter.display()