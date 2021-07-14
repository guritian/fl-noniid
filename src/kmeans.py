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
        pretrain_epoch = 50
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
        kmean = KMeans(n_clusters=10)
        kmean.fit(kmeans_weights)
        labels = kmean.labels_
        centers = kmean.cluster_centers_
        print("kmean: k={}, cost={}".format(10, int(kmean.score(kmeans_weights))))
        for c in range(10):
            cluster = kmeans_weights[labels == c]

# class K_Means(object):
#     # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
#     def __init__(self, k=2, tolerance=0.0001, max_iter=300):
#         self.k_ = k
#         self.tolerance_ = tolerance
#         self.max_iter_ = max_iter
#
#     def fit(self, data):
#         self.centers_ = {}
#         for i in range(self.k_):
#             self.centers_[i] = data[i]
#
#         for i in range(self.max_iter_):
#             self.clf_ = {}
#             for i in range(self.k_):
#                 self.clf_[i] = []
#             # print("质点:",self.centers_)
#             for feature in data:
#                 # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
#                 distances = []
#                 for center in self.centers_:
#                     # 欧拉距离
#                     # np.sqrt(np.sum((features-self.centers_[center])**2))
#                     distances.append(np.linalg.norm(feature - self.centers_[center]))
#                 classification = distances.index(min(distances))
#                 self.clf_[classification].append(feature)
#
#             # print("分组情况:",self.clf_)
#             prev_centers = dict(self.centers_)
#             for c in self.clf_:
#                 self.centers_[c] = np.average(self.clf_[c], axis=0)
#
#             # '中心点'是否在误差范围
#             optimized = True
#             for center in self.centers_:
#                 org_centers = prev_centers[center]
#                 cur_centers = self.centers_[center]
#                 if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
#                     optimized = False
#             if optimized:
#                 break
#
#     def predict(self, p_data):
#         distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
#         index = distances.index(min(distances))
#         return index


if __name__ == "__main__":
    #non-iid(1)   50round
    args = Parser().parse()
    print(args)
    test(args).run()
