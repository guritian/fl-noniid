"""
Adopted and modified from kuangliu's pytorch-cifar
[Source] https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet1(nn.Module):
    def __init__(self, num_of_classes):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, num_of_classes)

    def forward(self, x):
        with torch.no_grad():
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
