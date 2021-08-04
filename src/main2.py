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

class Sampler(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

    def __len__(self):
        return len(self.idxs)


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

class TotalLoss():
    def __init__(self,temp=2, lambda_old=0.1):
        self.kd_loss = KDLoss(temp=temp)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_old = lambda_old

    def __call__(self, preds, gts, old_preds=None, old_gts=None):
        preds_old, preds_new = preds[:, :], preds[:, :]

        old_task_loss = self.kd_loss(preds_old, old_gts)
        new_task_loss = self.ce_loss(preds_new, gts)

        total_loss = self.lambda_old * old_task_loss + new_task_loss
        return total_loss


def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

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


        #client1的模型
        model_new = LeNet(num_classes)

        # 每个client保存自己的网络参数
        device_weights = []
        for i in range(self.args.num_devices):
            device_weights.append(copy.deepcopy(model.state_dict()))
        #对client 0,1 进行预训练，使
        pretrain_epoch = self.args.pretrain_epoch
        for i in range(1):
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
        model.load_state_dict(device_weights[0])
        accuracy, loss, auc, kappa = Tester().test(
            test_dataset,
            0,
            0,
            copy.deepcopy(model),
            self.args
        )



        model_new.load_state_dict(device_weights[0])
        #清空old_model梯度
        model.zero_grad()
        new_fc3 = nn.Linear(168, num_classes)
        kaiming_normal_init(new_fc3)
        model_new.fc3.weight.data = new_fc3.weight.data
        model_new.fc3.bias.data = new_fc3.bias.data

        model_new = model_new.to(device)
        model = model.to(device)
        model_new.train()  # Train mode


        optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=args.sgd_momentum
            )
        #从client1 中提取数据训练
        dataloader = DataLoader(
            Sampler(train_dataset,
                    device_idxs[1]) if args.learning == "f" or args.learning == "fd" or args.learning == "f1" else train_dataset,
            batch_size=args.bs,
            shuffle=True
        )

        criterion = TotalLoss(temp=2)
        losses = []

        for epoch in range(200):
            batch_losses = []
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model_new.zero_grad()
                model.eval()
                with torch.no_grad():
                    old_outputs = model(data)

                output = model_new(data)
                #TODO  极有可能要修改
                loss = criterion(output, target,old_preds=output,old_gts=old_outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.learning == "f" or args.learning == "fd":
                    print(
                        f"\t Device {1} | Epoch {epoch + 1} | Batch {idx + 1} | Training loss: {loss.item()}")

                batch_losses.append(loss.item())

            avg_batch_loss = sum(batch_losses) / len(batch_losses)
            losses.append(avg_batch_loss)

            if args.learning == "f" or args.learning == "fd":
                print(
                    f"\t Device {1} | Epoch {epoch + 1} | Training loss: {avg_batch_loss} ")

        print(f"device 1  Testing:")
        accuracy1, loss, auc, kappa = Tester().test(
            test_dataset,
            1,
            1,
            copy.deepcopy(model),
            self.args
        )
        print(accuracy,accuracy1)


if __name__ == "__main__":
    #non-iid(1)   50round
    args = Parser().parse()
    print(args)
    FederatedLearning(args).run()
