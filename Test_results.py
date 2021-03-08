# -*- coding: utf-8 -*-
"""

Program to test the trained model and obtain the metrics.

Created on Sun Mar  7 11:23:17 2021

@author: alaga
"""

import os, glob, json
import torch, pickle
from sklearn.metrics import average_precision_score
import numpy as np
from evaluate_sceneseg import calc_ap, calc_miou, calc_precision_recall


data_dir = os.getcwd()
path = os.path.join(data_dir,"model")
model = torch.load(path)

data_dir = os.getcwd()

filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))

gt_dict = dict()
pr_dict = dict()
shot_to_end_frame_dict = dict()

for fn in filenames:
    x = pickle.load(open(fn, "rb"))
    
    ground_truth = x['scene_transition_boundary_ground_truth']
    start_label = torch.tensor([0])
    ground_truth = torch.cat((start_label, ground_truth), dim = 0)
    gt_dict[x["imdb_id"]] = ground_truth
    
    X1 = x['place']
    X2 = x['cast']
    X3 = x['action']
    X4 = x['audio']
    
    pred_labels = model(X1, X2, X3, X4)
    
    pr_dict[x["imdb_id"]] = pred_labels.squeeze(1).data.numpy()
    shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]


scores = dict()
scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)

scores["Miou"], _ = calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict)
scores["Precision"], scores["Recall"], scores["F1"], *_ = calc_precision_recall(gt_dict, pr_dict)

print("Scores:", json.dumps(scores, indent=4))



