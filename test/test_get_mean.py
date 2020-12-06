
"""
   Test get_mean_function

   Version 1 Abhinav Kumar 2019-06-01
"""
import sys
sys.path.insert(0, './pylib')
from SpatialMean import get_spatial_mean

import torch
from torch.autograd import Variable

x = -torch.ones(2,3,2,2)
x[0,0,0] = torch.ones(2,)
x = x.cuda()
x = Variable(x)
print(x)
print(get_spatial_mean(x))
