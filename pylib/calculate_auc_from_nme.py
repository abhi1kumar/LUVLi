

"""
    Sample Run:
    python pylib/calculate_auc_from_nme -i exp_dir/does_not_exist.npy

    Calculates the AUC as reported in ICCV'2017 and NeurIPS 2018 paper using the
    numpy file of all the test images.

    Version 2 2019-07-13 Abhinav Kumar Support for multi_pie added
    Version 1 2019-06-15 Abhinav Kumar Argparse and automatic selection of thresholds based on name of the input file
    Version 0 2019-06-14 Wenxuan Mou
"""

import argparse
import numpy as np
from sklearn import metrics
import sys

y_top = 100

def calculate_auc(xList, yList, nme_n, y_top):
    """
        This function was given by Wenxuan
    """
    auc = 0.0
    all_area = nme_n * y_top
    if len([i for i in xList if i == nme_n]) == 0:  # no element in x equals to nme_n(7)
        # if the maximum element in xList is smaller than nme_n
        if np.max(xList) < nme_n:
            auc = metrics.auc(np.asarray(xList), np.asarray(yList)) + (nme_n - np.max(xList))*y_top
            auc = auc *1.0/ all_area
        # if the maximum element in xList is larger than nme_n
        else:
            n_left = len([i for i in xList if i < nme_n])
            x_left = xList[n_left - 1]
            x_right = xList[n_left]
            y_left = yList[n_left - 1]
            y_right = yList[n_left]

            dx1 = nme_n - x_left
            dx2 = x_right - nme_n

            y_nme_n = dx1 * (y_right - y_left) / (dx1+dx2) + y_left
            xList_new = xList[:n_left]
            xList_new.append(nme_n)
            yList_new = yList[:n_left]
            yList_new.append(y_nme_n)

            xArr = np.asarray(xList_new)
            yArr = np.asarray(yList_new)

            auc = metrics.auc(xArr, yArr)
            auc = auc *1.0/ all_area
    else:
        n = len([i for i in xList if i <= nme_n])
        xArr = np.asarray(xList[:n])
        yArr = np.asarray(yList[:n])
        auc = metrics.auc(xArr,yArr)/all_area

    return auc*100.

def plot_cuc(nme_per_img):# similar to figure 8 in ICCV'17
    order_nme = np.sort(nme_per_img)
    n_img = len(order_nme)
    x = order_nme
    y = np.arange(1, n_img+1)*1.0/n_img

    return x, y

def get_failure_rate(nme_per_image, threshold_in_percent):
    """
        Calculates the failure rate as mentioned in the paper
        http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.pdf
        Failure Rate is defined as follows. 
        For one image, if NME is larger than a thresh-old, then it is considered a failed prediction
    """
    threshold = threshold_in_percent/100.0
    count = np.sum(nme_per_image > threshold)
    frate = float(count)/nme_per_image.shape[0]

    return frate*100.

#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--input'      , help = 'path of the input numpy file which contains the individual NMEs from the model', default='nme_per_image.npy')
ap.add_argument    ('-n', '--split_name' , help = 'name of the split', default='new_split. Used for WFLW')
ap.add_argument    ('-t', '--threshold'  , help = 'threshold in percent (default: 7)', type= float, default= 8)
args = ap.parse_args()

nme_n = [args.threshold]
nme_saved = np.load(args.input) #nme_all is the NME of all images

if "box" in args.input:
    # box in filename suggests it uses ground truth boxes as the normalizing term.
    nme_name    = "box"
else:
    nme_name    = "interocular"


if (nme_saved.shape[0] == 689):
    # Start index is decided based on Full, Common and Challenge
    # First two datasets are LFPW and Helen and have 554 images in total
    # Last dataset is IBUG and has 135 images
    start_index = [0, 554, 0]
    end_index   = [554, 689, 689]
    names_list  = ["Common", "Challenge", "Full"]
elif(nme_saved.shape[0] == 600):
    start_index = [0]     #[0  , 300,   0]
    end_index   = [600]   #[300, 600, 600]
    names_list  = ["Test"]#["Indoor", "Outdoor", "Test"]
elif(nme_saved.shape[0] == 6679):
    start_index = [0]
    end_index   = [6679]
    names_list  = ["menpo"]
elif(nme_saved.shape[0] == 812):
    start_index = [0]
    end_index   = [812]
    names_list  = ["multi_pie"]
elif("wflw" in args.input or "aflw" in args.input or 'cofw_68' in args.input):
    start_index = [0]
    end_index   = [nme_saved.shape[0]]
    names_list  = [args.split_name]
else:
    print("Some unknown file!!! Aborting")
    sys.exit(0)

for i in range(len(nme_n)):
    for j in range(len(names_list)):
        nme_all      = nme_saved[start_index[j]: end_index[j]]

        mean_error   = np.mean(nme_all)*100.
        x_gau, y_gau = plot_cuc(nme_all) 

        xList = list(x_gau * y_top)
        yList = list(y_gau * y_top)
        auc_test = calculate_auc(xList, yList, nme_n[i], y_top)
        frate    = get_failure_rate(nme_saved, nme_n[i])
        print("Dataset= {:25s} \tNME_threshold= {}% NME_{:11s}= {:.2f}% AUC= {:.2f}% FRate= {:.2f}%".format(names_list[j], nme_n[i], nme_name, mean_error, auc_test, frate))
