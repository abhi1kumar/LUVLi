from matplotlib import pyplot as plt
import numpy as np

from scipy import stats

"""
class your_distribution(stats.rv_continuous):
    def _pdf(self, x, y):
        return 1/(2*np.pi) *np.exp(-np.sqrt(np.power(x,2)))

distribution = your_distribution()
temp = distribution.rvs(size=100)
plt.hist(temp[0])
plt.show()
plt.close()
plt.hist(temp[1])
plt.show()
"""
# Implementation of Metropolis Hastings
"""
def density1(z):
    z = np.reshape(z, [z.shape[0], 2])
    z1, z2 = z[:, 0], z[:, 1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - np.log(exp1 + exp2)
    return np.exp(-u)

def metropolis_hastings(target_density, size=100):
    burnin_size = 10000
    size += burnin_size
    x0 = np.array([[0, 0]])
    xt = x0
    samples = []
    for i in range(size):
        xt_candidate = np.array([np.random.multivariate_normal(xt[0], np.eye(2))])
        accept_prob = (target_density(xt_candidate))/(target_density(xt))
        if np.random.uniform(0, 1) < accept_prob:
            xt = xt_candidate
        samples.append(xt)
    samples = np.array(samples[burnin_size:])
    samples = np.reshape(samples, [samples.shape[0], 2])
    return samples

samples = metropolis_hastings(density1)
plt.hexbin(samples[:,0], samples[:,1], cmap='rainbow')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
"""

"""
num_points_new = 1000
r1 = np.random.exponential(size=num_points_new)
print(r1.shape)
theta1 = np.random.uniform(0,2*np.pi,num_points_new)
print(theta1.shape)

x1 = np.multiply(r1, np.cos(theta1))
y1 = np.multiply(r1, np.sin(theta1))

plt.hist(theta1)
plt.show()
"""

def sample_from_simplified_laplacian(num_points):
    x = np.arange(-15,15,0.02)
    num = x.shape[0]
    prob = np.zeros((num,num))

    # Generate probability density first
    for i in range(num):
        for j in range(num):
            prob[i,j] = np.exp(-np.sqrt(x[i]**2 + x[j]**2))/(2*np.pi)
    prob = prob/np.sum(prob)
    prob = prob.flatten()

    t = np.arange(0,x.shape[0]**2)
    out = np.random.choice(t, num_points, p= prob)
    data = np.zeros((num_points, 2))

    for i in range(num_points):
        data[i,0] = x[out[i]//num]
        data[i,1] = x[out[i]%num]

    return data

num_points = 100000
data = sample_from_simplified_laplacian(num_points)
plt.hist2d(data[:, 0], data[:, 1], bins= 100, normed=True)
plt.show()
