import torch
from torchsummary import summary
import torch.nn as nn

model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 204))
print(model)
model = model.cuda()

summary(model,(1,2048))
