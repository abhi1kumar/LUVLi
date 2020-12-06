

"""
    Sample Run:
    python pylib/calculate_nme_uncetainty_on_landmarks.py -f abhinav_model_dir/run_62_evaluate/cofw_68 --laplacian
    python pylib/calculate_nme_uncetainty_on_landmarks.py -f abhinav_model_dir/run_5000_evaluate_old/aflw_ours_all --use_heatmaps

    Calculates the nme and uncertainty on external occluded points

    Version 0 2019-11-11 Abhinav Kumar
"""

import os, sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
from sklearn import metrics

from pylib.Cholesky import *
from plot.CommonPlottingOperations import *

means_rel            = "means.npy"
cholesky_rel         = "cholesky.npy"
ground_rel           = "ground_truth.npy"
visibility_rel       = "vis_gt.npy"
visibility_estimated = "vis_estimated.npy"

def get_nme_on_landmarks(prd, gt):
    nme_lmark = np.zeros((prd.shape[0], prd.shape[1]))
    for i in range(prd.shape[0]):
        bbox_d = compute_bboxd(gt[i])
        
        for j in range(prd.shape[1]):
            nme            = np.linalg.norm(prd[i,j] - gt[i,j])
            nme_lmark[i,j] = nme/bbox_d
            
    return nme_lmark
            
#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', default= 'abhinav_model_dir/run_62_evaluate/cofw_68', help= 'input folder relative path')
ap.add_argument(      '--laplacian', action = 'store_true'      , help= 'use laplacian likelihood instead of Gaussian')
ap.add_argument(      '--use_heatmaps' , action = 'store_true'      , help= 'whether cholesky is there or not')
args = ap.parse_args()

folder  = args.folder
is_covar = not args.use_heatmaps
is_vis   = not args.use_heatmaps

means   = np.load(os.path.join(folder, means_rel))
if is_covar:
    L_vect  = np.load(os.path.join(folder, cholesky_rel))
ground  = np.load(os.path.join(folder, ground_rel))
vis_gd     = np.load(os.path.join(folder, visibility_rel))
if is_vis:
    vis_est    = np.load(os.path.join(folder, visibility_estimated))

# First check NME is correct
d = compute_scale(ground)
nme_lmark = get_nme_on_landmarks(means, ground)

if is_covar:
    covar, det_sigma = cholesky_to_covar(L_vect, laplacian= args.laplacian)

    _, covar_norm = normalize_input(covar, d*d)
    sqrt_det_norm = np.sqrt(covar_norm[:,:,0,0]*covar_norm[:,:,1,1] - covar_norm[:,:,0,1]*covar_norm[:,:,1,0])
    four_det_norm = np.sqrt(sqrt_det_norm) 

thresh = 0.5
print("Landmark statistics")
if vis_gd is None:
    print("No visibility ground file")
    vis_classes = [1]
else:
    # flatten the vis_gd since we will be using the landmarks
    vis_gd    = vis_gd.flatten()
    if is_vis:
        vis_est   = vis_est.flatten()
    else:
        vis_est   = -1.0 * np.ones(vis_gd.shape)
    nme_lmark = nme_lmark.flatten()
    if is_covar:
        sqrt_det_sigma = np.sqrt(det_sigma.flatten())
        four_det_sigma = np.sqrt(np.sqrt(det_sigma.flatten()))
        sqrt_det_norm  = sqrt_det_norm.flatten()
        four_det_norm  = four_det_norm.flatten()
    else:
        sqrt_det_sigma = -1.0*np.ones(nme_lmark.shape)
        four_det_sigma = -1.0*np.ones(nme_lmark.shape)
        sqrt_det_norm  = -1.0*np.ones(nme_lmark.shape)
        four_det_norm  = -1.0*np.ones(nme_lmark.shape)

    vis_classes = np.unique(vis_gd).astype(int)
    vis_gt_bin  = np.zeros((vis_gd.shape))
    vis_pred_bin= np.zeros((vis_gd.shape))
    vis_pred_bin[vis_est>= thresh] = 1
    vis_pred_bin[vis_est < thresh] = 0
    
    for i in range(vis_classes.shape[0]):
        idx       = (vis_gd ==  vis_classes[i])
        num       = np.sum(idx)
        vis_class = np.mean(vis_est[idx])
        if vis_classes[i] <= 0:
            nme_class = -0.01
            vis_gt_bin[idx] = 0
        else:
            vis_gt_bin[idx] = 1
            nme_class      = np.mean(nme_lmark     [idx])
        acc_class = np.sum(vis_gt_bin[idx] == vis_pred_bin[idx])/float(num)
        sqrt_det_class      = np.mean(sqrt_det_sigma[idx])
        four_det_class      = np.mean(four_det_sigma[idx])
        sqrt_det_norm_class = np.mean(sqrt_det_norm [idx])
        four_det_norm_class = np.mean(four_det_norm [idx])
    
        print("Vis_Class= {},       Num= {:6d},       Acc= {:.2f}, Mean_vis_est= {:.2f}, NME_box        = {:.2f}%, Sqrt_Det= {:.2f}, Four_Det= {:.2f}, Sqrt_Det_Norm= {:.6f}, Four_Det_Norm= {:.6f}".format(vis_classes[i], num, acc_class, vis_class, nme_class*100.0, sqrt_det_class, four_det_class, sqrt_det_norm_class, four_det_norm_class))

# Do finaly for all visible
idx            = (vis_gd > 0)
num            = np.sum(idx)
vis_class      = np.mean(vis_est[idx])
acc_class      = np.sum(vis_gt_bin[idx] == vis_pred_bin[idx])/float(num)
nme_class      = np.mean(nme_lmark[idx])
sqrt_det_class      = np.mean(sqrt_det_sigma[idx])
four_det_class      = np.mean(four_det_sigma[idx])
sqrt_det_norm_class = np.mean(sqrt_det_norm [idx])
four_det_norm_class = np.mean(four_det_norm [idx])
print("Vis= All vis,       Num= {:6d},       Acc= {:.2f}, Mean_vis_est= {:.2f}, NME_box        = {:.2f}%, Sqrt_Det= {:.2f}, Four_Det= {:.2f}, Sqrt_Det_Norm= {:.6f}, Four_Det_Norm= {:.6f} ".format(num, acc_class, vis_class, nme_class*100.0, sqrt_det_class, four_det_class, sqrt_det_norm_class, four_det_norm_class))
