# DGL & Pytorch implementation of Enhanced Graph Embedding with Side information (EGES)

## Version
dgl==0.6.1, torch==1.9.0

## Paper
Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba: 

https://arxiv.org/pdf/1803.02349.pdf

https://arxiv.org/abs/1803.02349

## How to run
Create folder named `data`. Download two csv files from [here](https://github.com/Wang-Yu-Qing/dgl_data/tree/master/eges_data) into the `data` folder.

Run command: `python main.py` with default configuration, and the following message will shown up:

```
Using backend: pytorch
Num skus: 33344, num brands: 3662, num shops: 4785, num cates: 79
Epoch 00000 | Step 00000 | Step Loss 0.9117 | Epoch Avg Loss: 0.9117
Epoch 00000 | Step 00100 | Step Loss 0.8736 | Epoch Avg Loss: 0.8801
Epoch 00000 | Step 00200 | Step Loss 0.8975 | Epoch Avg Loss: 0.8785
Evaluate link prediction AUC: 0.6864
Epoch 00001 | Step 00000 | Step Loss 0.8695 | Epoch Avg Loss: 0.8695
Epoch 00001 | Step 00100 | Step Loss 0.8290 | Epoch Avg Loss: 0.8643
Epoch 00001 | Step 00200 | Step Loss 0.8012 | Epoch Avg Loss: 0.8604
Evaluate link prediction AUC: 0.6875
...
Epoch 00029 | Step 00000 | Step Loss 0.7095 | Epoch Avg Loss: 0.7095
Epoch 00029 | Step 00100 | Step Loss 0.7248 | Epoch Avg Loss: 0.7139
Epoch 00029 | Step 00200 | Step Loss 0.7123 | Epoch Avg Loss: 0.7134
Evaluate link prediction AUC: 0.7084
```

The AUC of link-prediction task on test graph is computed after each epoch is done.

## Reference
https://github.com/nonva/eges

https://github.com/wangzhegeek/EGES.git
