
import numpy as np
import matplotlib.pyplot as plt

def sample_from_bounded_gaussian(x):
    return max(-2*x, min(2*x, np.random.randn()*x))

sigma = 25
num = 1000000
arr = np.zeros((num,))
"""

arr = np.zeros((num,))

for i in range(num):
    arr[i] = sample_from_bounded_gaussian(100)

plt.hist(arr, bins=30)
plt.ylabel('Number')
plt.show()
plt.close()
"""
sigma = 0.25

for i in range(num):
    arr[i] = (2 ** (sample_from_bounded_gaussian(sigma)))

plt.hist(arr, normed=True, bins=30)
plt.ylabel('Number')
plt.grid()
plt.savefig('scale_1.png')
plt.close()

sigma = 0.125
for i in range(num):
    arr[i] = 1- sample_from_bounded_gaussian(sigma)

plt.hist(arr, normed=True, bins=30)
plt.ylabel('Number')
plt.grid()
plt.savefig('scale_2.png')
plt.close()
