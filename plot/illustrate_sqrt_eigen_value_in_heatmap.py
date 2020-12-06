

"""
	Sample Run
    python plot/illustrate_sqrt_eigen_value_in_heatmap.py
"""

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
from CommonPlottingOperations import *

my_dpi = 150
upsampling = 30.0
dim = 64
sigma = 3
k = 2
useless_lw = 6

cov = [[sigma, 0.8], [0.8, 0.25]]
mean= [31.5, 31.5]

xi = np.arange(32-k*sigma+1, 32+k*sigma-2, 1.0/upsampling)
xn, yn = np.meshgrid(xi, xi)
pos = np.empty(xn.shape + (2,))
pos[:, :, 0] = xn
pos[:, :, 1] = yn 
rv = multivariate_normal.pdf(pos, mean= mean, cov= cov);
print(rv.shape)


# Set up figure
#fig=plt.figure(figsize=(float(image.shape[0])/my_dpi,float(image.shape[1])/my_dpi),dpi=my_dpi)
fig=plt.figure(figsize=(float(xi.shape[0])/10,float(xi.shape[0])/10), dpi= my_dpi)
ax=plt.gca()

# Remove whitespace from around the image
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

# Set the gridding interval: here we use the major tick interval
myInterval=upsampling
loc = plticker.MultipleLocator(base=myInterval)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

# Add the grid
ax.grid(which='major', axis='both', linestyle='-', linewidth= useless_lw+2, color= "white")
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)

# Add the image
plt.imshow(rv   , cmap=cm.jet)


mean_x = xi.shape[0]/2.0
mean_y = mean_x

plot_mean_and_ellipse( cov, mean_x, mean_y, n=2, nstd=upsampling, ax=ax, color1= "dimgray", color2= "gray", markersize= useless_lw-1, linewidth= useless_lw+4, marker= 'o', markeredgewidth= 3)
#plt.plot(mean_x, mean_y, marker= "x", color= "black", markersize= 15, markeredgewidth= 5)


eVa, eVe = np.linalg.eig(cov)
scale = upsampling
cnt = 0
for e, v in zip(eVa, eVe.T):
	if cnt == 0:
		color='dimgray'
		linestyle='dashed'
		linewidth = useless_lw + 4
		zorder = 10
	else:
		color= 'black'
		linestyle='-'
		linewidth = useless_lw*2 + 2
		zorder = 11
	cnt += 1
	plt.plot([mean_x, mean_x + scale*np.sqrt(e)*v[0]], [mean_y, mean_y + scale*np.sqrt(e)*v[1]], color= color, linestyle=linestyle, lw= linewidth, zorder= zorder)


#plt.show()
plt.savefig("images/sqrt_eigen_value_heatmap.png")
