

import matplotlib.pyplot as plt
import numpy as np

num = 5
input = np.arange(-num, num, 0.1)

output1 = np.maximum(input, 0) + 0.00001
output2 = np.maximum(input, 0) + np.minimum((np.exp(input) - 1), 0) + 1
output3 = np.maximum(input, 0) + np.minimum(2*(np.exp(input) - 1), 0) + 2.

plt.plot(input, output1, label= "relu")
plt.plot(input, output2, label= "elu")
plt.plot(input, output3, label= "elu scale=2")
plt.grid(True)
plt.legend()
plt.show()
