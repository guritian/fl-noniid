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
import sklearn
from sklearn.cluster import MeanShift,DBSCAN
from sklearn.manifold import TSNE


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

        flag = False
        if flag:
            # 对每个client进行预训练，使
            pretrain_epoch = 300

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
                model_name = "model/client"+str(i)+".pth"
                torch.save(obj=weights,f=model_name)
        else:
            for i in range(self.args.num_devices):
                model_name = "model/client" + str(i) + ".pth"
                device_weights[i] = torch.load(model_name)

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

        # ms = MeanShift()
        # ms.bandwidth = 3.54
        #bandwidth = sklearn.cluster.estimate_bandwidth(kmeans_weights, quantile=0.3, n_samples=10, random_state=0)
        # 得到Mean-Sift方法
        # for i in range(100):
        #     ms.bandwidth = ms.bandwidth+0.01
        #     ms.fit(kmeans_weights)
        #     labels1 = ms.labels_
        #     labels_unique = np.unique(labels1)
        #     n_clusters_ = len(labels_unique)
        #     if n_clusters_ <=10:
        #         print(i)
        #         break
        # ms.bandwidth = ms.bandwidth + 0.01
        # ms.fit(kmeans_weights)
        # labels1 = ms.labels_
        # labels_unique = np.unique(labels1)
        # n_clusters_ = len(labels_unique)
        # # ms.bandwidth = 3
        # # ms.fit(kmeans_weights)
        # # labels1 = ms.labels_
        # # labels_unique = np.unique(labels1)
        # # n_clusters_ = len(labels_unique)
        # client_list_by_label1 = [[] for i in range(n_clusters_)]
        # for i in range(self.args.num_devices):
        #     # 在相应数据类别的列表中 加入设备index
        #     client_list_by_label1[labels1[i]].append(i)
        # print(client_list_by_label1)

        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2)
        tsne_weights = tsne.fit_transform(kmeans_weights)

        x_min, x_max = tsne_weights.min(0), tsne_weights.max(0)
        X_norm = (tsne_weights - x_min) / (x_max - x_min)  # 归一化









        ms1 = DBSCAN(eps=0.1,min_samples=5,metric='euclidean')
        ms1.fit(X_norm)
        labels2 = ms1.labels_
        labels_unique = np.unique(labels2)
        n_clusters_ = len(labels_unique)

        color_list = []

        client_list_by_label2 = [[] for i in range(n_clusters_)]
        for i in range(self.args.num_devices):
            # 在相应数据类别的列表中 加入设备index
            client_list_by_label2[labels2[i]].append(i)
            color_list.append(labels2[i])
        print(client_list_by_label2)

        fig, ax = plt.subplots()

        scatter = ax.scatter(X_norm[:, 0], X_norm[:, 1], c=color_list)
        for i in range(X_norm.shape[0]):
            #plt.scatter(X_norm)
            plt.text(X_norm[i, 0], X_norm[i, 1], str(i), #color=plt.cm.Set1(y[i]),
                     fontdict={'size': 6})
        # legend1 = ax.legend(*scatter.legend_elements(),
        #                     loc="lower right", title="classes")
        # ax.add_artist(legend1)
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

        plt.show()

        # for i in range(X_norm.shape[0]):
        #     #plt.scatter(X_norm)
        #     plt.text(X_norm[i, 0], X_norm[i, 1], str(client_labels[i]), #color=plt.cm.Set1(y[i]),
        #              fontdict={'weight': 'bold', 'size': 9})
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()



if __name__ == "__main__":
    #non-iid(1)   50round

    args = Parser().parse()
    print(args)
    test(args).run()
