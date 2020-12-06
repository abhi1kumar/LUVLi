
"""
    Test KL Divergence
    Version 1 Abhinav Kumar 2019-06-09

"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


kl_div   = nn.KLDivLoss(reduce=False) # For pytorch>= 0.4.0 use reduction=None

# KL Divergence in Torch expects input to be log-probabilities while target to be probabilities only.
# https://pytorch.org/docs/stable/nn.html#torch.nn.KLDivLoss
#   
# P(x/N) =      1                         ___-1 
#         _____________  exp [ -(x-\mu)^T \    (x-\mu)  ]
#                 ___        [            /__           ]
#          2 \pi |\   |      [ ________________________ ]
#                |/__ |      [            2 \pi         ]
#
# Ignoring the constant scaling terms, we get
#                     ___                 ___-1
# log P(x/N) = -log( |\   | ) - (x-\mu)^T \    (x-\mu)  = log_prob_1 + log_prob_2
#                    |/__ |               /__

target = Variable(torch.from_numpy(np.array([[0.0, 0, 1], [0.0, 0, 1]])))
input  = Variable(torch.from_numpy(np.array([[0.333, 0.333, 0.333], [0.1, 0.2, 0.7]])))

temp   = torch.log(input)
print(torch.log(input))

#print(np.log10(input.data.numpy()))
#print(np.log10(target.data.numpy()))

print(torch.__version__)

out = kl_div(temp, target) 
print(out)
