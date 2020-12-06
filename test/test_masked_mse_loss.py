



import os, sys
sys.path.insert(0, os.getcwd())

import torch
from torch.autograd import Variable
import torch.nn as nn

from loss.masked_mse_loss import *

loss = MaskedMSELoss()
loss2 = nn.MSELoss()

a = torch.Tensor([ [  [[1,2,3],[4,5,6]], [[11,12,13],[21,22,23]]  ], [  [[1, 2, 3], [4, 5, 6]], [[27, 28, 29], [30, 31, 32.]] ] ])
b = torch.zeros(a.shape)

print("With mask")
mask = torch.Tensor([[1,0],[1,0]])
t = loss(Variable(a), Variable(b), Variable(mask))
print(t.data.numpy())
print("Expected= 15.166")


print("Without mask")
mask = torch.Tensor([[1,1],[1,1]])
t = loss(Variable(a), Variable(b), Variable(mask))
t2 = loss2(Variable(a), Variable(b))
print(t.data.numpy())
print("Expected= {}".format(t2.data.numpy()))

print("\n")

# Test with only mean points
a = torch.Tensor([  [[1,2],[3,4],[5,6]], [[11,12],[13,21],[22,23]]  ])
b = torch.zeros(a.shape)

print("With mask")
mask = torch.Tensor([[1,1,1],[0,0,0]])
t = loss(Variable(a), Variable(b), Variable(mask))
print(t.data.numpy())
print("Expected= 7.58")

print("Without mask")
mask = torch.Tensor([[1,1,1],[1,1,1]])
t = loss(Variable(a), Variable(b), Variable(mask))
t2 = loss2(Variable(a), Variable(b))
print(t.data.numpy())
print("Expected= {}".format(t2.data.numpy()))
