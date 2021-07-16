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
from sklearn.cluster import MeanShift

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log')

class test():
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
            # torch.cuda.set_device(self.args.gpu)

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
        train_dataset, test_dataset, device_idxs, client_labels = utils.get_dataset_dist(self.args)
        for i in range(self.args.num_devices):
            print(str(i)+" "+str(client_labels[i]))

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

        # Optimization technique
        if self.args.warmup_model:
            weights = utils.warmup_model(model, train_dataset, test_dataset, device, self.args)
            model.load_state_dict(weights)

        avg_weights_diff = []
        global_train_losses = []
        global_test_losses = []
        global_accuracies = []
        global_aucs = []
        global_kappas = []
        # 用来进行最后的 各个client的fine tuning
        final_weight = []

        # 每个client保存自己的网络参数
        device_weights = []
        #用于kmeans 的 展开网络参数
        kmeans_weights = []
        for i in range(self.args.num_devices):
            device_weights.append(copy.deepcopy(model.state_dict()))
        # 对每个client进行预训练，使
        pretrain_epoch = 200
        for i in range(self.args.num_devices):
            weights, loss = Trainer().pre_train(
                epoch=pretrain_epoch,
                dataset=train_dataset,
                idxs=device_idxs[i],
                device_num=i,
                device=device,
                model=copy.deepcopy(model),  # Avoid continuously training same model on different devices
                args=self.args
            )
            device_weights[i] = weights

        for i in range(self.args.num_devices):
            first = True
            for param_tensor in device_weights[i]:
                numpy_para = device_weights[i][param_tensor].cpu().numpy()
                numpy_para = numpy_para.reshape(-1)
                if first:
                    transform_feature = numpy_para
                    first = False
                else:
                    transform_feature = np.append(transform_feature,numpy_para)
                #print(transform_feature.shape)
            kmeans_weights.append(transform_feature)
        # kmean = KMeans(n_clusters=5)
        # kmean.fit(kmeans_weights)
        # labels = kmean.labels_
        # centers = kmean.cluster_centers_
        # client_list_by_label = [[] for i in range(5)]
        # for i in range(self.args.num_devices):
        #     # 在相应数据类别的列表中 加入设备index
        #     client_list_by_label[labels[i]].append(i)
        #print(client_list_by_label)

        ms = MeanShift()
        # 得到Mean-Sift方法
        ms.fit(kmeans_weights)
        labels1 = ms.labels_
        labels_unique = np.unique(labels1)
        n_clusters_ = len(labels_unique)
        client_list_by_label1 = [[] for i in range(n_clusters_)]
        for i in range(self.args.num_devices):
            # 在相应数据类别的列表中 加入设备index
            client_list_by_label1[labels1[i]].append(i)
        print(client_list_by_label1)


if __name__ == "__main__":
    #non-iid(1)   50round
    args = Parser().parse()
    print(args)
    test(args).run()
