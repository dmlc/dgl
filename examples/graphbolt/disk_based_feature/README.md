## Overview

This project demonstrates how to use GraphBolt to train and evaluate a GraphSAGE model for node classification task on large graphs, where node features are on-disk and fetched using `DiskBasedFeature`. GraphBolt utilizes [S3-FIFO](https://s3fifo.com) to cache frequently required features and io_uring to fetch cache-missed features from disk.

# Node classification Task

This example demonstrates how to run node classification task with **GraphBolt.DiskBasedFeature**.

## Run on `ogbn-papers100M` dataset

This instruction trains on three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 10GB CPU cache, using neighbor sampling.

* Dataset:

|     Dataset     | Graph Size | Feature Size | Feature Dim |
| :-------------: | :--------: | :----------: | :---------: |
| Ogbn-papers100M |   13 GB   |    53 GB    |     128     |

* Instruction:

```
python node_classification.py --num-gpu-cached-features=0 --num-cpu-cached-features=20000000 --batch-dependency=128 --cpu-feature-cache-policy=s3-fifo --dataset=ogbn-papers100M --sample-mode=sample_neighbor --epochs=3
```

## Results

Below results are collected on an AWS EC2 g5.8xlarge instance with 128GB RAM, 32 cores, an 24GB A10G GPU and a instance storage of 250K IOPS.

```
Training: 1178it [04:33,  4.31it/s, num_nodes=669013, gpu_cache_miss=1, cpu_cache_miss=0.0876]
Evaluating: 123it [00:25,  4.92it/s, num_nodes=628067, gpu_cache_miss=1, cpu_cache_miss=0.0865]
Epoch 00, Loss: 1.4147, Approx. Train: 0.5788, Approx. Val: 0.6179, Time: 273.57775235176086s
Training: 1178it [03:11,  6.15it/s, num_nodes=668114, gpu_cache_miss=1, cpu_cache_miss=0.0745]
Evaluating: 123it [00:24,  5.02it/s, num_nodes=626141, gpu_cache_miss=1, cpu_cache_miss=0.0744]
Epoch 01, Loss: 1.1454, Approx. Train: 0.6381, Approx. Val: 0.6438, Time: 191.6689577102661s
Training: 1178it [03:10,  6.19it/s, num_nodes=663826, gpu_cache_miss=1, cpu_cache_miss=0.0703]
Evaluating: 123it [00:24,  5.02it/s, num_nodes=628172, gpu_cache_miss=1, cpu_cache_miss=0.0703]
Epoch 02, Loss: 1.0980, Approx. Train: 0.6507, Approx. Val: 0.6599, Time: 190.37866187095642s
```
