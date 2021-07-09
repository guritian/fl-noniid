from torchvision import datasets, transforms
import math
import random
import torch
import copy
import csv
from datetime import datetime
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from trainer import Tester


class Utils():
    def get_dataset_dist(self, args):
        if args.dataset == "mnist":
            data_dir = "./data/mnist"

            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_dataset = datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )
        elif args.dataset == "cifar10":
            data_dir = "./data/cifar10"

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )

        if args.learning == "f" or args.learning == "fd" or args.learning == "fd":
            if args.iid:
                user_idxs = self.iid_dist(train_dataset, args)
            else:
                user_idxs = self.noniid_dist(train_dataset, args)
        else:
            user_idxs = []

        return train_dataset, test_dataset, user_idxs

    def iid_dist(self, dataset, args):
        data_per_device = math.floor(len(dataset)/args.num_devices)
        idxs = list(range(len(dataset)))    # Put all data index in list
        users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(idxs)
        for i in range(args.num_devices):
            users_idxs[i] = idxs[i*data_per_device:(i+1)*data_per_device]
            if args.max_data_per_device:
                users_idxs[i] = users_idxs[i][0:args.max_data_per_device]
                
        return users_idxs

    def noniid_dist(self, dataset, args):
        if args.class_per_device:   # Use classes per device
            if args.class_per_device > len(dataset.classes):
                raise OverflowError("Class per device is larger than number of classes")

            if args.equal_dist:
                raise NotImplementedError("Class per device can only be used with unequal distributions")
            else:
                current_classs = 0
                users_classes = [[] for i in range(args.num_devices)]  # Classes dictionary for devices
                classes_devives = [[] for i in range(len(dataset.classes))]  # Devices in each class

                # Distribute class numbers to devices
                for i in range(args.num_devices):
                    next_current_class = (current_classs+args.class_per_device)%len(dataset.classes)
                    if next_current_class > current_classs:
                        users_classes[i] = np.arange(current_classs, next_current_class)
                    else:
                        users_classes[i] = np.append(
                            np.arange(current_classs, len(dataset.classes)),
                            np.arange(0, next_current_class)
                        )
                    
                    for j in users_classes[i]:
                        classes_devives[j].append(i)
                    
                    current_classs = next_current_class

                # Combine indexes and labels for sorting
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

                users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

                current_idx = 0
                for i in range(len(dataset.classes)):
                    if not len(classes_devives[i]):
                        continue

                    send_to_device = 0
                    for j in range(current_idx, len(idxs_labels[0])):
                        if idxs_labels[1, j] != i:
                            current_idx = j
                            break

                        users_idxs[classes_devives[i][send_to_device]].append(idxs_labels[0, j])
                        send_to_device = (send_to_device+1)%len(classes_devives[i])

                if args.max_data_per_device:
                    for i in range(args.num_devices):
                        users_idxs[i] = users_idxs[i][0:args.max_data_per_device]
                """
                # Validate results
                tmp_list = []
                for i in range(args.num_devices):
                    tmp_list = list(set(tmp_list) | set(users_idxs[i])) 
                print(len(tmp_list))

                sum = 0
                for i in range(args.num_devices):
                    sum += len(users_idxs[i])
                print(sum)
                """

                return users_idxs

        else:   # Use non-IIDness
            if args.equal_dist:
                data_per_device = math.floor(len(dataset)/args.num_devices)
                users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

                # Combine indexes and labels for sorting
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
                idxs = idxs_labels[0, :].tolist()

                niid_data_per_device = int(data_per_device*args.noniidness/100)

                # Distribute non-IID data
                for i in range(args.num_devices):
                    users_idxs[i] = idxs[i*niid_data_per_device:(i+1)*niid_data_per_device]

                # Still have some data
                if args.num_devices*niid_data_per_device < len(dataset):
                    # Filter distributed data
                    idxs = idxs[args.num_devices*niid_data_per_device:]
                    # Randomize data after sorting
                    random.shuffle(idxs)

                    remaining_data_per_device = data_per_device-niid_data_per_device

                    # Distribute IID data
                    for i in range(args.num_devices):
                        users_idxs[i].extend(idxs[i*remaining_data_per_device:(i+1)*remaining_data_per_device])
                
                """
                # Validate results
                for i in users_idxs[0]:
                    print(idxs_labels[1, np.where(idxs_labels[0] == i)])

                sum = 0
                for i in range(args.num_devices):
                    sum += len(users_idxs[i])
                print(sum)
                """
                if args.max_data_per_device:
                    for i in range(args.num_devices):
                        users_idxs[i] = users_idxs[i][0:args.max_data_per_device]
                
                return users_idxs
            else:
                # Max data per device
                max = math.floor(len(dataset)/args.num_devices)
                # Each device get [0.2*max, max) amount of data
                data_per_device = [int(random.uniform(max/5, max)) for i in range(args.num_devices)]

                users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

                # Combine indexes and labels for sorting
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
                idxs = idxs_labels[0, :].tolist()

                niid_data_per_device = [int(data_per_device[i]*args.noniidness/100) for i in range(args.num_devices)]

                current_idx = 0
                # Distribute non-IID data
                for i in range(args.num_devices):
                    users_idxs[i] = idxs[current_idx:current_idx+niid_data_per_device[i]]
                    current_idx += niid_data_per_device[i]

                # Filter distributed data
                idxs = idxs[current_idx:]
                # Randomize data after sorting
                random.shuffle(idxs)

                remaining_data_per_device = [data_per_device[i]-niid_data_per_device[i] for i in range(args.num_devices)]

                current_idx = 0
                # Distribute IID data
                for i in range(args.num_devices):
                    users_idxs[i].extend(idxs[current_idx:current_idx+remaining_data_per_device[i]])
                    current_idx += remaining_data_per_device[i]

                if args.max_data_per_device:
                    for i in range(args.num_devices):
                        users_idxs[i] = users_idxs[i][0:args.max_data_per_device]

                """
                # Validate results
                tmp_list = []
                for i in range(args.num_devices):
                    tmp_list = list(set(tmp_list) | set(users_idxs[i])) 
                print(len(tmp_list))

                sum = 0
                for i in range(args.num_devices):
                    sum += len(users_idxs[i])
                print(sum)
                """

                return users_idxs

    # Non-weighted averaging...
    def fed_avg(self, weights):
        w = copy.deepcopy(weights[0])   # Weight from first device
        for key in w.keys():
            for i in range(1, len(weights)):    # Other devices
                w[key] += weights[i][key]   # Sum up weights
            w[key] = torch.div(w[key], len(weights))    # Get average weigh ts
        return w

    def cal_avg_weight_diff(self, weights_list, avg_weights):
        w = copy.deepcopy(weights_list)
        w2 = copy.deepcopy(avg_weights)

        key = list(w2.keys())

        for key in list(w2.keys()):
            w2[key] = w2[key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

        # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
        diff_list = []  
        print("\n\tWeight difference:")

        for key in list(w2.keys()):
            tmp2 = []
            for i in range(len(w)):
                tmp = []
                w[i][key] = w[i][key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

                for j in range(len(w[i][key])):
                    tmp.append(abs((w[i][key][j]-w2[key][j])/w2[key][j]))   # Abs((val(device)-val(average))/val(average))

                average = sum(tmp)/len(tmp) # Calculate average
                tmp2.append(average) 
                print(f"\t\tWeight difference | Weight {i + 1} | {key} | {average}")

            average = sum(tmp2)/len(tmp2) # Calculate average
            diff_list.append(average) 

        return sum(diff_list)/len(diff_list)

    def save_results_to_file(self, args, avg_weights_diff,
                             global_train_losses, global_test_losses,
                             global_accuracies, aucs,
                             kappas):
        iid = "iid" if args.iid else "niid"

        # Create folder if it doesn't exist
        if not os.path.exists("results"):
            os.mkdir("results")

        if args.model == 'lenetBN':
            fl_algo = 'fedBN'
        else:
            fl_algo = 'fedAvg'

        f = open(f"./results/{datetime.now()}_{args.dataset}_{args.model}_{iid}_{args.noniidness}_{args.equal_dist}_{args.class_per_device}_{fl_algo}_{args.num_devices}_{args.frac}.csv", "w")
        #f = open(f"d:\\village\\fl-noniid\\src\\results\\{args.dataset}_{args.model}_{iid}_{args.noniidness}_{args.equal_dist}_{args.class_per_device}_{fl_algo}_{args.num_devices}_{args.frac}.csv",
            #"w")
        with f:
            if args.learning == "f" or args.learning == "fd":
                fnames = ["round", "average weight differences",
                        "train losses", "test losses", "test accuracies", 
                        "average AUCs", "Kappas"]
            else:
                fnames = ["round", "train losses", 
                        "test losses", "test accuracies", 
                        "average AUCs", "Kappas"]
            writer = csv.DictWriter(f, fieldnames=fnames)

            writer.writeheader()

            for i in range(len(global_train_losses)):
                if args.learning == "f" or args.learning == "fd":
                    writer.writerow({
                        "round": i+1,
                        "average weight differences": avg_weights_diff[i],
                        "train losses": global_train_losses[i],
                        "test losses": global_test_losses[i],
                        "test accuracies": global_accuracies[i],
                        "average AUCs": aucs[i],
                        "Kappas": kappas[i]
                    })
                else:
                    writer.writerow({
                        "round": i+1,
                        "train losses": global_train_losses[i],
                        "test losses": global_test_losses[i],
                        "test accuracies": global_accuracies[i],
                        "average AUCs": aucs[i],
                        "Kappas": kappas[i]
                    })
        print(f"Results stored in results/{args.dataset}_{args.optim}_{iid}_{args.noniidness}_{args.equal_dist}_{args.class_per_device}.csv")

    def warmup_model(self, model, train_dataset, test_dataset, device, args):
        model.to(device)
        model.train()   # Train mode

        print("Training warmup model...")

        if args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr, 
                momentum=args.sgd_momentum
            )
        elif args.optim == "adagrad":
            optimizer = torch.optim.Adagrad(
                model.parameters(), 
                lr=args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.lr
            )

        dataloader = DataLoader(
            train_dataset,
            batch_size=args.bs, 
            shuffle=True
        )
        
        loss_function = nn.CrossEntropyLoss().to(device)


        for idx, (data, target) in enumerate(dataloader):
            model.train()   # Train mode

            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            if idx % 40 == 0:
                accuracy, loss = Tester().test(
                    test_dataset, 
                    -1, 
                    device, 
                    copy.deepcopy(model), 
                    args
                )

                print(f"Accuracy reaches {accuracy}")

                if accuracy >= 0.6:
                    break

        print(f"Trained warmup model to accuracy {accuracy}!")

        return copy.deepcopy(model.state_dict())


    #def plot_result(self,args):

        # Non-weighted averaging...
    def fed_avg(self, weights):
        w = copy.deepcopy(weights[0])  # Weight from first device
        for key in w.keys():
            for i in range(1, len(weights)):  # Other devices
                w[key] += weights[i][key]  # Sum up weights
            w[key] = torch.div(w[key], len(weights))  # Get average weigh ts
        return w

##TODO  有机会的话 把client_weights 修改为 根据样本数量进行计算（因为目前所有client训练数据规模一样）
    def communication(args, frac,local_weights):

        w = copy.deepcopy(local_weights[0])
        with torch.no_grad():

            if frac == 1:
                for key in w.keys():
                    if 'bn' not in key:
                        for i in range(1, len(local_weights)):  # Other devices
                            w[key] += local_weights[i][key]  # Sum up weights
                        w[key] = torch.div(w[key], len(local_weights))  # Get average weigh ts
            else:
                for key in w.keys():
                    for i in range(1, len(local_weights)):  # Other devices
                        w[key] += local_weights[i][key]  # Sum up weights
                    w[key] = torch.div(w[key], len(local_weights))  # Get average weigh ts


            #将 各个本地client的非BN层参数修改为
            for key in w.keys():
                if 'bn' not in key:
                    for i in range(0, len(local_weights)):
                        local_weights[i][key] = w[key]
        return w
