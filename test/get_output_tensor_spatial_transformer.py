import torch
from torchsummary import summary
import torch.nn as nn
from torch.autograd import Variable

model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#model = model.cuda()

input = Variable(torch.zeros((1, 1, 28, 28)).float())
output = model(input)
print(output.data.shape)


model = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 7, stride= 2, padding= 3),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True),
            nn.Conv2d(in_channels= 8, out_channels= 10, kernel_size= 5, stride= 2, padding=0),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True)
        )

input = Variable(torch.zeros((1, 1, 64, 64)).float())
output = model(input)
print(output.data.shape)
