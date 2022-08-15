# Check the IoU and tangent angle for the images that the inference is run on.

import json
import os

iou_path = '/home/Evaluation-of-Frame-Field-Learning-using-Orthophotos/data/PrivateDataset/raw/test/images/poly_shapefile.simple.tol_1/aggr_iou.json'
metrics_path = '/home/Evaluation-of-Frame-Field-Learning-using-Orthophotos/data/PrivateDataset/raw/test/images/poly_shapefile.simple.tol_1/aggr_metrics.json'

with open(iou_path) as iou:
    data = json.load(iou)
print('IoU : ', data, '\n')

with open(metrics_path) as angle:
    ang = json.load(angle)
    
metrics = ang.get('max_angle_diffs')
avg = sum(metrics)/len(metrics)

print('Tangent angle: ', avg)