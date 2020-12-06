

import os, sys
sys.path.insert(0, os.getcwd())

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

c = np.load('cholesky.npy')
sigma, det_cov = cholesky_to_covar(c)
print(sigma.shape)

# Bring it down to 64 x 64 pixels
sigma = sigma/float(16)


minor_axis = np.zeros((sigma.shape[0] * sigma.shape[1], ))
cnt = 0
for i in range(sigma.shape[0]):
    for j in range(sigma.shape[1]):
        w, v = LA.eig(sigma[i, j])
        minor_axis[cnt] = np.sqrt(np.min(w))
        cnt += 1

plt.figure()
plt.hist(minor_axis, density= True, bins= 100)
plt.ylabel('Probability')
plt.grid(True)
plt.title('Distribution of min eigen value of covariance matrix') 
plt.savefig('images/distribution_minor_axis.png')
