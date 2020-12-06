

"""
    Sample Run:
    python plot/plot_visibility_roc_and_precision_recall.py -i run_5004_evaluate

    Plots precision recall curve and roc curve on visibilities and also gives the optimal threshold

    Version 1 2020-03-07 Abhinav Kumar
"""
import os, sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from CommonPlottingOperations import *

from sklearn.metrics import precision_recall_curve, roc_curve

dodge_blue = np.array([0.12, 0.56, 1.0])
dpi   = 200
alpha = 0.9
msize = 10
fs    = 16
lw    = 3
delta = 0.01
matplotlib.rcParams.update({'font.size': fs})
figsize = (6, 6)

#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id'    , default= 'run_5004_evaluate', help= 'input folder relative path')
args   = ap.parse_args()

folder_path = os.path.join("abhinav_model_dir", os.path.join(args.exp_id, "aflw_ours_all"))
print("Folder= {}".format(folder_path))
ground_truth, means, cholesky, vis_gt, vis_estimated = load_all(folder_path, load_vis= True, load_images= False)

# Get Precision Recall
# Convert multi_class to binary class
vis_gt[vis_gt >= 1] = 1
vis_gt = vis_gt.flatten()
vis_estimated = vis_estimated.flatten()

precision, recall, thresholds = precision_recall_curve(vis_gt, vis_estimated)
fpr, tpr, _                   = roc_curve(vis_gt, vis_estimated)

index = np.where(precision-recall == 0)
opt_threshold = thresholds[index][0]
print("Threshold at which precision equals recall = {:.2f}".format(opt_threshold))

temp = np.arange(0,1 + delta,delta)

fig = plt.figure(figsize= figsize, dpi= dpi)
plt.plot(fpr, tpr, color= "dodgerblue", label= "ROC curve", linewidth= lw)
plt.xlim((0-delta, 1))
plt.ylim((0, 1+delta))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title ('ROC Curve. Threshold = {:.2f}'.format(opt_threshold))
plt.grid(True)
savefig(plt, "images/visibility_roc.png", tight_flag= True)

fig = plt.figure(figsize= figsize, dpi= dpi)
plt.plot(recall, precision, color= "dodgerblue", label= "PR curve", linewidth= lw)
#plt.plot(temp, temp, color='k', lw= 0.75)
plt.xlim((0, 1+delta))
plt.ylim((0.8, 1+delta))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title ('PR Curve. Threshold = {:.2f}'.format(opt_threshold))
plt.grid(True)
savefig(plt, "images/visibility_precision_recall.png", tight_flag= True)
