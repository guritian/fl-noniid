# Research on Federated Learning with non-IID data

Federated Learning, a new decentralized collaborative Machine Learning framework presented by Google in 2017, enables end devices to collaboratively learn a shared Machine Learning model, while reducing privacy and security risks, attaining lower latency and providing personalized models to clients, but it suffers from performance issue with non-IID data distribution. This project is a comprehensive study of the phenomenon and related work. The results show that precision loses during averaging step due to the property of averaging and serious parameter divergence. Different training settings are tested to evaluate the effects on performance.

## Setup
```pip3 install -r requirements.txt```

## Sample output
```
python3 main.py --iid=0 --round=1 --equal_dist=0 --noniidness=20 

[skipped]

Losses on training data:
        [2.28713748959511]

Losses on testing data:
        [2.294593244601207]

Accuracies on testing data:
        [0.0981]

Average AUCs on testing data:
        [0.5000564597341957]

Kappas on testing data:
        [0.00011084257451980585]

Final accuracy: 0.0981
Final loss: 2.294593244601207

Results stored in results/2020-01-21 14:20:57.905349_mnist_sgd_niid_20.0_0_0.csv
```

## Files

Program arguments: [arg_parser](/src/arg_parser.py)

Report: [report.pdf](/report.pdf)

