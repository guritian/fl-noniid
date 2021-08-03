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
        #用来进行最后的 各个client的fine tuning
        final_weight = []

        # 每个client保存自己的网络参数
        device_weights = []
        for i in range(self.args.num_devices):
            device_weights.append(copy.deepcopy(model.state_dict()))
        # #对每个client进行预训练，使
        # pretrain_epoch = self.args.pretrain_epoch
        # for i in range(self.args.num_devices):
        #     weights, loss = Trainer().pre_train(
        #         epoch = pretrain_epoch,
        #         dataset = train_dataset,
        #         idxs = device_idxs[i],
        #         device_num = i,
        #         device = device,
        #         model = copy.deepcopy(model),  # Avoid continuously training same model on different devices
        #         args = self.args
        #     )
        #     device_weights[i] = weights


        #赋一个初值
        avg_weights = device_weights[0]
        for round in tqdm(range(self.args.round)):
            # Train step
            print(f"\nRound {round + 1} Training:")

            local_weights = []
            local_losses = []


            #用于 计数  所有类别是否都加入到联邦学习中
            #TODO 这只是一个比较粗略的方案  真实情况下 不可能事先知道 所有类别
            #用来判断包含数据分类总数是否达标
            #class_set = set()

            # Select fraction of devices (minimum 1 device)


            train_devices = random.sample(
                range(self.args.num_devices),
                max(1, int(self.args.num_devices * self.args.frac))
            )



            print(f"\tDevices selected: {[x + 1 for x in train_devices]}\n")

            # Train on each device and return weights and loss
            for device_num in train_devices:
                #第一轮时  都采用预训练的local weight进行训练
                if round == 0:
                    model.load_state_dict(device_weights[device_num])
                else:
                    model.load_state_dict(avg_weights)
                weights, loss = Trainer().train(
                    train_dataset,
                    device_idxs[device_num],
                    round,
                    device_num,
                    device,
                    copy.deepcopy(model),  # Avoid continuously training same model on different devices
                    self.args
                )

                local_weights.append(weights)
                local_losses.append(loss)

            avg_weights = utils.fed_avg(local_weights)  # Federated averaging
            for device_num in train_devices:
                device_weights[device_num] = avg_weights
            model.load_state_dict(avg_weights)  # Load new weights

            if self.args.cal_para_diff:
                avg_weight_diff = utils.cal_avg_weight_diff(local_weights, avg_weights)
            else:
                avg_weight_diff = 0
            avg_loss = sum(local_losses) / len(local_losses)

            if self.args.cal_para_diff:
                print(f"\n\tRound {round + 1} | Average weight difference: {avg_weight_diff}")

            print(f"\tRound {round + 1} | Average training loss: {avg_loss}\n")

            global_train_losses.append(avg_loss)
            avg_weights_diff.append(avg_weight_diff)

            # Test step
            print(f"Round {round + 1}  Testing:")
            accuracy, loss, auc, kappa = Tester().test(
                test_dataset,
                round,
                device,
                copy.deepcopy(model),
                self.args
            )

            scalar_name = 'accuracy_avg_'+str(self.args.round)+"_"+str(self.args.class_per_device)+"_lr"+str(self.args.lr)
            writer.add_scalar(scalar_name,
                              accuracy,
                              round)

            print(f"\tRound {round + 1} | Average accuracy: {accuracy}")
            print(f"\tRound {round + 1} | Average testing loss: {loss}\n")
            print(f"\tRound {round + 1} | Average AUC: {auc}")
            print(f"\tRound {round + 1} | Kappa: {kappa}\n")

            global_test_losses.append(loss)
            global_accuracies.append(accuracy)
            global_aucs.append(auc)
            global_kappas.append(kappa)

            #如果整个fedavg round已经结束了
            if(round+1 == self.args.round ):
                final_weight = avg_weights

            # Quit early if satisfy certain situations
            if accuracy >= self.args.train_until_acc / 100:
                print(
                    f"Accuracy reached {self.args.train_until_acc / 100} in round {round + 1}, stopping...")
                break
            if self.args.stop_if_improvment_lt:
                if round > 0:
                    if global_accuracies[-2] + self.args.stop_if_improvment_lt / 100 >= global_accuracies[-1]:
                        break

        # Write results to file
        if self.args.save_results:

            utils.save_results_to_file(
                self.args,
                avg_weights_diff,
                global_train_losses,
                global_test_losses,
                global_accuracies,
                global_aucs,
                global_kappas
            )
        filename = str(self.args.round)+"_"+str(args.class_per_device)+"_lr"+str(self.args.lr)
        f = open("./results/"+filename+"_fed_avg.txt","w")
        f.writelines(str(global_accuracies))
        f.close()


if __name__ == "__main__":
    #non-iid(1)   50round
    args = Parser().parse()
    print(args)
    FederatedLearning(args).run()
