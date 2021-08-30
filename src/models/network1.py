"""
Adopted and modified from kuangliu's pytorch-cifar
[Source] https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn as nn
import torch.nn.functional as F


class network1(nn.Module):
    def __init__(self, old_num_of_classes,new_num_of_classes):
        super(network1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3_old = nn.Linear(168, old_num_of_classes)
        self.fc3_new = nn.Linear(168,new_num_of_classes)

    def forward(self, x,model=2):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))


        #得到新模型的预测值
        if model == 1:
            out = self.fc3_new(out)
        else:
            out = self.fc3_old(out)
        out = F.log_softmax(out,dim=1)
        return out
