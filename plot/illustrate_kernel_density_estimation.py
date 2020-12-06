

"""
  Sample Run
  python plot/illustrate_kernel_density_estimation.py
  
  Taken from
  https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/ 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.interpolate import griddata

fs = 22
plt.rcParams.update({'font.size': fs})


def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    #kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x), var_type='c', **kwargs)
    #return kde.pdf(x_grid)
    kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

"""
# The grid we'll use for plotting
x_grid = np.linspace(0, 64, 64)

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
x_obs = np.concatenate([norm(32, 1.).rvs(400), norm(48, 0.3).rvs(100)])
pdf_true = (0.8 * norm(32, 1).pdf(x_grid) + 0.2 * norm(48, 0.3).pdf(x_grid))

xv, yv = np.meshgrid(x_grid, x_grid)
xv = xv.reshape((1,-1))
yv = yv.reshape((1,-1))
x_grid2 = np.vstack((xv,yv))

xt, yt = np.random.multivariate_normal([32,32], 1*np.eye(2), 200).T 
x_obs2 = np.vstack((xt.reshape((1,-1)),yt.reshape((1,-1))))

# Plot the three kernel density estimates
fig = plt.figure(figsize=(8, 4))
pdf_2 = kde_statsmodels_m(x_obs2, x_grid2, bandwidth=1)
print(pdf_2.shape)
print(x_grid2.shape)
pdf_2 = pdf_2.reshape((x_grid.shape[0],-1))

plt.imshow(pdf_2)
#plt.imshow(
#plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
#plt.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
plt.title("Multi variate KDE")
plt.xlim(0, 64)
plt.show()

"""
def np_bivariate_normal_pdf(domain, mean, step, std= 1, min_val= 0):
  X = np.arange(min_val, domain, step)
  Y = np.arange(min_val, domain, step)
  X, Y = np.meshgrid(X, Y)
  R = np.sqrt((X-mean[0])**2 + (Y-mean[1])**2)
  Z = ((1. / np.sqrt(2 * np.pi* std)) * np.exp(-.5*R**2/std**2))
  return X, Y, Z
  
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plt_plot_bivariate_normal_pdf(x, y, z, name="blah"):
  fig = plt.figure(figsize=(12, 6), dpi= 150)
  ax = fig.gca(projection='3d')
  ax.plot_surface(x, y, z, 
                  cmap=cm.jet,
                  linewidth=0,
                  rstride=1,
                  cstride=1, 
                  antialiased=True)
  ax.set_xlabel('x', labelpad=15)
  ax.set_ylabel('y', labelpad=15)
  ax.set_zlabel('z', labelpad=15);
  ax.view_init(elev=40, azim=-10)
  print("Saving to {}".format(name))
  #plt.show()
  plt.savefig(name, bbox_inches = 'tight')
  plt.close()
  
def get_shifted_gaussians(shift=6, weight=0.15):
    _, _, z2 = np_bivariate_normal_pdf(64, (32-shift,32-shift), sampling, std)
    _, _, z3 = np_bivariate_normal_pdf(64, (32-shift,32+shift), sampling, std)
    _, _, z4 = np_bivariate_normal_pdf(64, (32+shift,32-shift), sampling, std)
    _, _, z5 = np_bivariate_normal_pdf(64, (32+shift,32+shift), sampling, std)
    _, _, z6 = np_bivariate_normal_pdf(64, (32-1.414*shift,32), sampling, std)
    _, _, z7 = np_bivariate_normal_pdf(64, (32+1.414*shift,32), sampling, std)
    _, _, z8 = np_bivariate_normal_pdf(64, (32,32+1.414*shift), sampling, std)
    _, _, z9 = np_bivariate_normal_pdf(64, (32,32-1.414*shift), sampling, std)
    z_temp = weight*(z2 + z3 + z4 + z5 + z6 + z7 + z8 + z9)
    
    return z_temp

def get_shifted_heatmap(heatmap, shift=6, weight=0.15):

    heatmap[32-shift,32-shift] += weight
    heatmap[32-shift,32+shift] += weight
    heatmap[32+shift,32-shift] += weight
    heatmap[32+shift,32+shift] += weight
    heatmap[32-int(1.414*shift),32] += weight
    heatmap[32+int(1.414*shift),32] += weight
    heatmap[32,32-int(1.414*shift)] += weight
    heatmap[32,32+int(1.414*shift)] += weight
        
    return heatmap
 
sampling = 0.25
std = 3 
scale = 10

wt = [0.4, 0.16, 0.08]
# Heatmap
heatmap = np.zeros((64,64))
heatmap[32,32] = wt[0]

heatmap = get_shifted_heatmap(heatmap, shift=8 , weight=wt[1])
heatmap = get_shifted_heatmap(heatmap, shift=16, weight=wt[2])

xi = np.arange(0, 64)
xig, yig = np.meshgrid(xi, xi)
plt_plot_bivariate_normal_pdf(xig, yig, heatmap, name='images/kernel_dummy_heatmap.png')

X, Y, z1 = np_bivariate_normal_pdf(32, (0,0), sampling, std, min_val= -32)
plt_plot_bivariate_normal_pdf(X, Y, z1, name= "images/kernel_gaussian.png")


X, Y, z1 = np_bivariate_normal_pdf(64, (32,32), sampling, std)
zshift1 = get_shifted_gaussians(shift=8,  weight=wt[1])
zshift2 = get_shifted_gaussians(shift=16, weight=wt[2])
z_final = wt[0]*z1 + zshift1 + zshift2
plt_plot_bivariate_normal_pdf(X, Y, z_final, name= "images/kernel_density.png")

"""
xi = np.arange(X.min(), X.max(), (X.max()-X.min())/(X.shape[0]*scale))
yi = np.arange(Y.min(), Y.max(), (Y.max()-Y.min())/(Y.shape[0]*scale))
xig, yig = np.meshgrid(xi, yi)
zi = griddata(np.hstack((X.reshape((-1,1)), Y.reshape((-1,1)) )), z_final.flatten(), (xig, yig), method='cubic')
print(zi.shape)
plt_plot_bivariate_normal_pdf(xig, yig, zi)

from scipy import interpolate
xnew, ynew = np.mgrid[0:64:sampling/4., 0:64:sampling/4.]
tck = interpolate.bisplrep(X, Y, z_final, s=0)
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
plt_plot_bivariate_normal_pdf(xnew, ynew, znew)
"""

