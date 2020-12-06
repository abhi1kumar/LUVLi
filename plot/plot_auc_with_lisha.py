

import os,sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from CommonPlottingOperations import *
from pylib.Cholesky import *

dodge_blue = np.array([30, 144, 255])/255.
dpi   = 200
fs    = 18
lw    = 3
color1 = (1,0.45,0.45)
color2 = "dodgerblue"
matplotlib.rcParams.update({'font.size': fs})

folder   = "abhinav_model_dir/run_940_evaluate/300W_test"
nme_ours = np.load(os.path.join(folder, "nme_new_box_per_image.npy"))

lisha_folder = os.path.join('plot_uncertainty_NeurlPS', 'result_new')
ground       = np.load(os.path.join(lisha_folder, 'twtest64_gt.npy'))
means        = np.load(os.path.join(lisha_folder, 'kdng.npy'))
nme_norm     = get_err(means, ground)

nme_norm_lisha  = nme_norm.reshape((600, 68))
nme_image_lisha = np.mean(nme_norm_lisha, 1)

# Now plot
x_lisha, y_lisha = plot_cuc(nme_image_lisha)
x_ours, y_ours = plot_cuc(nme_ours)

plt.figure(figsize=(9.6,6), dpi =dpi)
plt.plot(x_ours ,y_ours , color= color2, label= "UGLLI (Ours)", lw= lw)
plt.plot(x_lisha,y_lisha, color= color1, label= "KDN-Gaussian (Chen et al)", lw= lw)
plt.grid(True)
plt.legend(loc="center right")
plt.xlabel(r'$NME$ Threshold')
plt.ylabel(r"Detection Rate (Fraction correct)")
plt.xlim((0, 0.1))
plt.xticks(np.arange(0.0, 0.11, 0.01))

index = np.where(x_ours <= 0.07)
# Sometimes we do not get the value of 0.07
maxindex = np.max(index) + 1
# Insert the next index to be 0.07 which will be the average of
# previous and next values.
# This is used for plotting only not for AUC calculation
x_ours[maxindex] = 0.07 
y_ours[maxindex] = 0.5*(y_ours[maxindex-1] + y_ours[maxindex+1])

index = np.where(x_ours <= 0.07)
plt.fill_between(x_ours[index],y_ours[index], color= color2, alpha= 0.6)
plt.savefig("images/auc_with_lisha.png")
plt.show()
plt.close()