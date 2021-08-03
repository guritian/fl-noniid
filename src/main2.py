import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")


import time
from arg_parser import Parser
from arg_parser1 import  Parser1
from src.models.lenet1 import LeNet1
from src.models.lenetBN import LeNetBN
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

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log')


class KDLoss():
    def __init__(self, temp=2):
        self.temp = temp
        self.log_sotfmax = nn.LogSoftmax(dim=-1)

    def __call__(self, preds, gts):
        preds = F.softmax(preds, dim=-1)
        preds = torch.pow(preds, 1. / self.temp)
        # l_preds = F.softmax(preds, dim=-1)
        l_preds = self.log_sotfmax(preds)


        gts = F.softmax(gts, dim=-1)
        gts = torch.pow(gts, 1. / self.temp)
        l_gts = self.log_sotfmax(gts)


        l_preds = torch.log(l_preds)
        l_preds[l_preds != l_preds] = 0.  # Eliminate NaN values
        loss = torch.mean(torch.sum(-l_gts * l_preds, axis=1))
        return loss

class FederatedLearning():
    def __init__(self, args):
        self.args = args

    def run(self):
        start = time.time()

        # Print arguments
        if self.args.verbose:
            print("Arguments:")
            print(f"\t{self.args}")

        # Set training on CPU/GPU
        device = "cpu"
        if self.args.gpu is not None:
            if torch.cuda.is_available():
                device = "cuda"
            #torch.cuda.set_device(self.args.gpu)

        # Set manual random seed
        if not self.args.random_seed:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            torch.backends.cudnn.deterministic = True

        utils = Utils()

        # Get dataset and data distribution over devices
        # TODO client_labels 目前用来表示每个client上的数据分类
        train_dataset, test_dataset, device_idxs,client_labels = utils.get_dataset_dist(self.args)

        # Get number of classes (MNIST: 10, CIFAR10: 10)
        if self.args.dataset == "mnist":
            num_classes = 10
        else:
            num_classes = 10

        # Set training model (VGG11/LeNet/ResNet18)
        if self.args.model == "vgg11":
            model = VGG("VGG11", num_classes)
        elif self.args.model == "lenet":
            model = LeNet(num_classes)
        else:
            model = ResNet18(num_classes)



        # 每个client保存自己的网络参数
        device_weights = []
        for i in range(self.args.num_devices):
            device_weights.append(copy.deepcopy(model.state_dict()))
        #对client 0,1 进行预训练，使
        pretrain_epoch = self.args.pretrain_epoch
        for i in range(2):
            weights, loss = Trainer().pre_train(
                epoch = pretrain_epoch,
                dataset = train_dataset,
                idxs = device_idxs[i],
                device_num = i,
                device = device,
                model = copy.deepcopy(model),  # Avoid continuously training same model on different devices
                args = self.args
            )
            device_weights[i] = weights







if __name__ == "__main__":
    #non-iid(1)   50round
    args = Parser().parse()
    print(args)
    FederatedLearning(args).run()
