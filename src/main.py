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


# class FederatedLearning():
#     def __init__(self, args):
#         self.args = args
#
#     def run(self):
#         start = time.time()
#
#         # Print arguments
#         if self.args.verbose:
#             print("Arguments:")
#             print(f"\t{self.args}")
#
#         # Set training on CPU/GPU
#         device = "cpu"
#         if self.args.gpu is not None:
#             if torch.cuda.is_available():
#                 device = "cuda"
#             torch.cuda.set_device(self.args.gpu)
#
#         # Set manual random seed
#         if not self.args.random_seed:
#             torch.manual_seed(42)
#             torch.cuda.manual_seed(42)
#             np.random.seed(42)
#             random.seed(42)
#             torch.backends.cudnn.deterministic = True
#
#         utils = Utils()
#
#         # Get dataset and data distribution over devices
#         train_dataset, test_dataset, device_idxs = utils.get_dataset_dist(self.args)
#
#         # Get number of classes (MNIST: 10, CIFAR10: 10)
#         if self.args.dataset == "mnist":
#             num_classes = 10
#         else:
#             num_classes = 10
#
#         # Set training model (VGG11/LeNet/ResNet18)
#         if self.args.model == "vgg11":
#             model = VGG("VGG11", num_classes)
#         elif self.args.model == "lenet":
#             model = LeNet(num_classes)
#         elif self.args.model == "lenetBN":
#             model = LeNetBN(num_classes)
#         else:
#             model = ResNet18(num_classes)
#
#         # Optimization technique
#         if self.args.warmup_model:
#             weights = utils.warmup_model(model, train_dataset, test_dataset, device, self.args)
#             model.load_state_dict(weights)
#
#         avg_weights_diff = []
#         global_train_losses = []
#         global_test_losses = []
#         global_accuracies = []
#         global_aucs = []
#         global_kappas = []
#
#
#         for round in tqdm(range(self.args.round)):
#             # Train step
#             print(f"\nRound {round+1} Training:")
#
#             local_weights = []
#             local_losses = []
#
#             # Select fraction of devices (minimum 1 device)
#             train_devices = random.sample(
#                 range(self.args.num_devices),
#                 max(1, int(self.args.num_devices*self.args.frac))
#             )
#
#             print(f"\tDevices selected: {[x+1 for x in train_devices]}\n")
#
#             # Train on each device and return weights and loss
#             for device_num in train_devices:
#                 weights, loss = Trainer().train(
#                     train_dataset,
#                     device_idxs[device_num],
#                     round,
#                     device_num,
#                     device,
#                     copy.deepcopy(model),   # Avoid continuously training same model on different devices
#                     self.args
#                 )
#
#                 local_weights.append(weights)
#                 local_losses.append(loss)
#
#             if args.model == 'lenet':
#                 avg_weights = utils.fed_avg(local_weights)# Federated averaging
#             elif args.model == 'lenetBN':
#                 avg_weights = utils.communication(local_weights)  # Federated averaging
#
#
#             model.load_state_dict(avg_weights)  # Load new weights
#
#             if self.args.cal_para_diff:
#                 avg_weight_diff = utils.cal_avg_weight_diff(local_weights, avg_weights)
#             else:
#                  avg_weight_diff = 0
#             avg_loss = sum(local_losses)/len(local_losses)
#
#             if self.args.cal_para_diff:
#                 print(f"\n\tRound {round+1} | Average weight difference: {avg_weight_diff}")
#
#             print(f"\tRound {round+1} | Average training loss: {avg_loss}\n")
#
#             global_train_losses.append(avg_loss)
#             avg_weights_diff.append(avg_weight_diff)
#
#
#
#             # Test step
#             print(f"Round {round+1}  Testing:")
#             accuracy, loss, auc, kappa = Tester().test(
#                 test_dataset,
#                 round,
#                 device,
#                 copy.deepcopy(model),
#                 self.args
#             )
#
#             print(f"\tRound {round+1} | Average accuracy: {accuracy}")
#             print(f"\tRound {round+1} | Average testing loss: {loss}\n")
#             print(f"\tRound {round+1} | Average AUC: {auc}")
#             print(f"\tRound {round+1} | Kappa: {kappa}\n")
#
#             global_test_losses.append(loss)
#             global_accuracies.append(accuracy)
#             global_aucs.append(auc)
#             global_kappas.append(kappa)
#
#
#
#
#             # Quit early if satisfy certain situations
#             if accuracy >= self.args.train_until_acc/100:
#                 print(
#                     f"Accuracy reached {self.args.train_until_acc/100} in round {round+1}, stopping...")
#                 break
#             if self.args.stop_if_improvment_lt:
#                 if round > 0:
#                     if global_accuracies[-2]+self.args.stop_if_improvment_lt/100 >= global_accuracies[-1]:
#                         break
#
#         end = time.time()
#         print(f"\nTime used: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")
#
#         # Print final results
#         if self.args.cal_para_diff:
#             print("\nAverage weight differences:")
#             print(f"\t{avg_weights_diff}\n")
#
#         print("Losses on training data:")
#         print(f"\t{global_train_losses}\n")
#         print("Losses on testing data:")
#         print(f"\t{global_test_losses}\n")
#         print("Accuracies on testing data:")
#         print(f"\t{global_accuracies}\n")
#         print("Average AUCs on testing data:")
#         print(f"\t{global_aucs}\n")
#         print("Kappas on testing data:")
#         print(f"\t{global_kappas}\n")
#
#         print(f"Final accuracy: {global_accuracies[-1]}")
#         print(f"Final loss: {global_test_losses[-1]}\n")
#
#         # Write results to file
#         if self.args.save_results:
#             utils.save_results_to_file(
#                 self.args,
#                 avg_weights_diff,
#                 global_train_losses,
#                 global_test_losses,
#                 global_accuracies,
#                 global_aucs,
#                 global_kappas
#             )


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
        #device = "cpu"
        device = "cuda"
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
        train_dataset, test_dataset, device_idxs = utils.get_dataset_dist(self.args)

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
        elif self.args.model == "lenetBN":
            model = LeNetBN(num_classes)
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

        #每个client保存自己的BN层参数
        device_weights = []
        for i in range(self.args.num_devices):
            device_weights.append(copy.deepcopy(model.state_dict()))


        for round in tqdm(range(self.args.round)):
            # Train step
            print(f"\nRound {round + 1} Training:")

            local_weights = []
            local_losses = []

            # Select fraction of devices (minimum 1 device)
            train_devices = random.sample(
                range(self.args.num_devices),
                max(1, int(self.args.num_devices * self.args.frac))
            )

            # #将local client BN层参数放进来
            # for x in train_devices:
            #     local_weights.append(device_weights[x])


            print(f"\tDevices selected: {[x + 1 for x in train_devices]}\n")

            # Train on each device and return weights and loss
            for device_num in train_devices:
                model.load_state_dict(device_weights[device_num])
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

            if args.model == 'lenet':
                avg_weights = utils.fed_avg(local_weights)  # Federated averaging
            elif args.model == 'lenetBN':
                if args.frac == 1:
                    avg_weights = utils.communication(args.frac,local_weights)# Federated BN
                else:
                    avg_weights = utils.communication(args.frac,local_weights)  # Federated BN

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
            i=0
            accuracy, loss, auc, kappa  = 0,0,0,0
            #由于加入了BN层 所以模型的测试 更改为每个local都进行测试
            for device_num in train_devices:

                model.load_state_dict(local_weights[i])
                device_weights[device_num] = local_weights[i]

                temp_accuracy, temp_loss, temp_auc, temp_kappa = Tester().test(
                    test_dataset,
                    round,
                    device,
                    copy.deepcopy(model),
                    self.args
                )
                i+=1
                print(f"\tDevice {device_num} Round {round + 1} | Average accuracy: {temp_accuracy}")
                print(f"\tDevice {device_num} Round {round + 1} | Average testing loss: {temp_loss}\n")
                print(f"\tDevice {device_num} Round {round + 1} | Average AUC: {temp_auc}")
                print(f"\tDevice {device_num} Round {round + 1} | Kappa: {temp_kappa}\n")
                accuracy += temp_accuracy
                loss += temp_loss
                auc += temp_auc
                kappa += temp_kappa
            accuracy /= len(local_weights)
            loss /= len(local_weights)
            auc /= len(local_weights)
            kappa /= len(local_weights)

            writer.add_scalar('accuracy',
                              accuracy,
                              round)
            global_test_losses.append(loss)
            global_accuracies.append(accuracy)
            global_aucs.append(auc)
            global_kappas.append(kappa)

            # Quit early if satisfy certain situations
            if accuracy >= self.args.train_until_acc / 100:
                print(
                    f"Accuracy reached {self.args.train_until_acc / 100} in round {round + 1}, stopping...")
                break
            if self.args.stop_if_improvment_lt:
                if round > 0:
                    if global_accuracies[-2] + self.args.stop_if_improvment_lt / 100 >= global_accuracies[-1]:
                        break

        end = time.time()
        print(f"\nTime used: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")

        # Print final results
        if self.args.cal_para_diff:
            print("\nAverage weight differences:")
            print(f"\t{avg_weights_diff}\n")

        print("Losses on training data:")
        print(f"\t{global_train_losses}\n")
        print("Losses on testing data:")
        print(f"\t{global_test_losses}\n")
        print("Accuracies on testing data:")
        print(f"\t{global_accuracies}\n")
        print("Average AUCs on testing data:")
        print(f"\t{global_aucs}\n")
        print("Kappas on testing data:")
        print(f"\t{global_kappas}\n")

        print(f"Final accuracy: {global_accuracies[-1]}")
        print(f"Final loss: {global_test_losses[-1]}\n")

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


class FederatedDecoupleLearning():
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
        train_dataset, test_dataset, device_idxs = utils.get_dataset_dist(self.args)

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

        for round in tqdm(range(self.args.round)):
            # Train step
            print(f"\nRound {round + 1} Training:")

            local_weights = []
            local_losses = []

            # Select fraction of devices (minimum 1 device)
            train_devices = random.sample(
                range(self.args.num_devices),
                max(1, int(self.args.num_devices * self.args.frac))
            )

            print(f"\tDevices selected: {[x + 1 for x in train_devices]}\n")

            # Train on each device and return weights and loss
            for device_num in train_devices:
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
        #对所有的client进行一轮 只训分类器
        modelByDecouple = LeNet1(num_classes)
        modelByDecouple.load_state_dict(final_weight)
        local_decouple_accuracy = []
        print(f"Final accuracy: {global_accuracies[-1]}")
        for device_num in range(10):
        # for device_num in range(self.args.num_devices):

            print(f"device {device_num}  FedAvg Testing:")
            #每个client先测一下使用FedAvg的准确率
            accuracy, loss, auc, kappa = Tester().test(
                test_dataset,
                round,
                device,
                copy.deepcopy(modelByDecouple),
                self.args
            )
            print(f"\tdevice {device_num} | Average accuracy: {accuracy}")
            print(f"\tdevice {device_num} | Average testing loss: {loss}\n")
            print(f"\tdevice {device_num} | Average AUC: {auc}")
            print(f"\tdevice {device_num} | Kappa: {kappa}\n")


            #每个Local Client都使用 FedAvg算法得到的最终模型参数进行 解耦分类器训练
            weights, loss = Trainer().train(
                train_dataset,
                device_idxs[device_num],
                round,
                device_num,
                device,
                copy.deepcopy(modelByDecouple),  # Avoid continuously training same model on different devices
                self.args
            )
            #对经过fine tune的 Local Client 进行测试
            model.load_state_dict(weights)
            # Test step
            print(f"device {device_num}  FDL Testing:")
            accuracy, loss, auc, kappa = Tester().test(
                test_dataset,
                round,
                device,
                copy.deepcopy(model),
                self.args
            )
            local_decouple_accuracy.append(accuracy)
            print(f"\tdevice {device_num} | Average accuracy: {accuracy}")
            print(f"\tdevice {device_num} | Average testing loss: {loss}\n")
            print(f"\tdevice {device_num} | Average AUC: {auc}")
            print(f"\tdevice {device_num} | Kappa: {kappa}\n")
            print("-----------------------------------------------------------------------")




        end = time.time()
        print(f"\nTime used: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")

        print(f"Final accuracy: {global_accuracies[-1]}")
        print(f"经过fine tuning 后各个Client的测试准确率：")
        print(local_decouple_accuracy)
        print(f"Final loss: {global_test_losses[-1]}\n")

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



class FederatedLearning1():
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
        #对每个client进行预训练，使
        pretrain_epoch = 50
        for i in range(self.args.num_devices):
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
        #对client进行聚类
        kmeans_weights = []
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
        client_list_by_label = [[] for i in range(self.args.class_num)]
        for i in range(self.args.num_devices):
            #在相应数据类别的列表中 加入设备index
            client_list_by_label[labels[i]].append(i)

        #赋一个初值
        avg_weights = device_weights[0]
        for round in tqdm(range(self.args.round)):
            # Train step
            print(f"\nRound {round + 1} Training:")

            local_weights = []
            local_losses = []

            train_devices = []
            #用于 计数  所有类别是否都加入到联邦学习中
            #TODO 这只是一个比较粗略的方案  真实情况下 不可能事先知道 所有类别
            #用来判断包含数据分类总数是否达标
            #class_set = set()

            # Select fraction of devices (minimum 1 device)
            #TODO 设置算法 挑选包含所有分类数据的devices （目前假设每个client上数据分类已知，在labels）
            #while len(class_set)<10:
            #     此注释方法选取client 通过真实的数据类别  没有考虑隐私性
            #     temp_train_devices = random.sample(
            #     range(self.args.num_devices),
            #     max(1, int(self.args.num_devices * self.args.frac))
            # )
            #     for client_index in temp_train_devices:
            #         #取出具体client的类别列表
            #         temp = set(client_labels[client_index])
            #         before_class_set_len = len(class_set)
            #         class_set = class_set|temp
            #         if(before_class_set_len != len(class_set)):
            #             train_devices.append(client_index)
            #根据聚类的结果 对client进行选择
            for i in range(self.args.class_num):
                random_index = random.choice(range(len(client_list_by_label[i])))
                train_devices.append(client_list_by_label[i][random_index])
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
            scalar = 'accuracy'+str(self.args.round)+"_"+str(self.args.class_per_device)+"_lr"+str(self.args.lr)
            writer.add_scalar(scalar,
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
        filename = str(self.args.round)+"_"+str(self.args.class_per_device)+"_lr"+str(self.args.lr)
        f = open("./results/"+filename+"_fed_cluster.txt","w")
        f.writelines(str(global_accuracies))
        f.close()





class CentralizedLearning():
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
            torch.cuda.set_device(self.args.gpu)

        # Set manual random seed
        if not self.args.random_seed:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            torch.backends.cudnn.deterministic = True

        utils = Utils()

        # Get dataset and data distribution over devices
        train_dataset, test_dataset, _ = utils.get_dataset_dist(self.args)

        # Get number of classes (MNIST: 10, CIFAR100: 10)
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

        train_losses = []
        test_losses = []
        accuracies = []
        aucs = []
        kappas = []

        if self.args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr, 
                momentum=self.args.sgd_momentum
            )
        elif self.args.optim == "adagrad":
            optimizer = torch.optim.Adagrad(
                model.parameters(), 
                lr=self.args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.args.lr
            )

        for epoch in tqdm(range(self.args.epoch)):
            # Train step
            print(f"Epoch {epoch+1} Training:")

            """
            model.to(device)
            model.train()   # Train mode

            dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.bs, 
                shuffle=True
            )
            
            loss_function = nn.CrossEntropyLoss().to(device)

            batch_losses = []
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                if not idx % 10:
                    print(f"\tEpoch {epoch+1} | {idx*self.args.bs}/{len(train_dataset)} | Training loss: {loss.item()}")

                batch_losses.append(loss.item())

            train_losses.append(sum(batch_losses)/len(batch_losses))

            print(f"\nEpoch {epoch+1} | Average training loss: {loss}\n")
            """

            weights, loss = Trainer().train(
                train_dataset, 
                0, 
                epoch, 
                0, 
                device, 
                copy.deepcopy(model),
                self.args
            )

            model.load_state_dict(weights)
            train_losses.append(loss)



            # Test step
            print(f"Epoch {epoch+1}  Testing:")
            accuracy, loss, auc, kappa = Tester().test(
                test_dataset, 
                epoch, 
                device, 
                model, 
                self.args
            )

            print(f"Epoch {epoch+1} | Accuracy: {accuracy}")
            print(f"Epoch {epoch+1} | Average testing loss: {loss}\n")
            print(f"Epoch {epoch+1} | Average AUC: {auc}")
            print(f"Epoch {epoch+1} | Kappa: {kappa}\n")

            test_losses.append(loss)
            accuracies.append(accuracy)
            aucs.append(auc)
            kappas.append(kappa)


            # Quit early if satisfy certain situations
            if accuracy >= self.args.train_until_acc/100:
                print(
                    f"Accuracy reached {self.args.train_until_acc/100} in epoch {epoch+1}, stopping...")
                break
            if self.args.stop_if_improvment_lt:
                if epoch > 0:
                    if accuracies[-2]+self.args.stop_if_improvment_lt/100 >= accuracies[-1]:
                        break

        end = time.time()
        print(f"\nTime used: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")

        # Print final results
        print("Losses on training data:")
        print(f"\t{train_losses}\n")
        print("Losses on testing data:")
        print(f"\t{test_losses}\n")
        print("Accuracies on testing data:")
        print(f"\t{accuracies}\n")
        print("Average AUCs on testing data:")
        print(f"\t{aucs}\n")
        print("Kappas on testing data:")
        print(f"\t{kappas}\n")

        print(f"Final accuracy: {accuracies[-1]}")
        print(f"Final loss: {test_losses[-1]}\n")

        # Write results to file
        if self.args.save_results:
            utils.save_results_to_file(
                self.args,  
                [],
                train_losses, 
                test_losses, 
                accuracies,
                aucs,
                kappas
            )
 
  
if __name__ == "__main__":
    #non-iid(1)   50round
    args = Parser().parse()
    print(args)
    if args.learning == "f1" :
        FederatedLearning1(args).run()
    elif args.learning == "f" :
        FederatedLearning1(args).run()
    elif args.learning == "fd":
        FederatedDecoupleLearning(args).run()
    else:
        CentralizedLearning(args).run()

    # non-iid(2)   50round
    # args.class_per_device = 2
    # FederatedLearning(args).run()