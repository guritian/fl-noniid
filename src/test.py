import math
import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")


import time
from arg_parser import Parser
from arg_parser1 import  Parser1
from src.models.lenet1 import LeNet1
from src.models.lenetBN import LeNetBN
from src.models.network1 import network1

from utils import Utils
import torch
import numpy as np
import random
from tqdm import tqdm
from trainer import Trainer, Tester
import copy
from models import *
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans


model = network1(2)
model_new = network1(4)

state_dict = model.state_dict()
state_dict = {k: v for k, v in state_dict.items() if 'fc3' not in k }
state_dict['fc3.weight']= model_new.fc3.weight
state_dict['fc3.bias']= model_new.fc3.bias
model_new.load_state_dict(state_dict)

# data = torch.rand(64,3,32,32)
# output = model(data)
# output1 = torch.zeros(64,2)
# output1[:,:2] = output
# loss_function = nn.CrossEntropyLoss().to("cuda")
# target = torch.randint(10,(64,))
# loss = loss_function(output1,target)
# print(loss)

data = torch.rand(64,3,32,32)
output = model(data)
output1 = torch.zeros(64,10)
output1[:,:] = -9e+9
