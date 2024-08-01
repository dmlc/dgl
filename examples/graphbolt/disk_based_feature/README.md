## Overview

This project demonstrates how to use GraphBolt to train and evaluate a GraphSAGE model for node classification task on large graphs, where node features are on-disk and fetched using `DiskBasedFeature`. GraphBolt utilizes various in-house implemented caching policy algorithms such as [SIEVE](https://cachemon.github.io/SIEVE-website/), [S3-FIFO](https://s3fifo.com), LRU and [CLOCK](https://people.csail.mit.edu/saltzer/Multics/MHP-Saltzer-060508/bookcases/M00s/M0104%20074-12%29.PDF) to cache frequently required features and io_uring to fetch cache-missed features from disk. The SIEVE algorithm is the default option.

# Node classification task

This example demonstrates how to run node classification task with **GraphBolt.DiskBasedFeature**.

## Run on `ogbn-papers100M` dataset

|     Dataset     | Graph Size | Feature Size | Feature Dim |
| :-------------: | :--------: | :----------: | :---------: |
| Ogbn-papers100M |   13 GB   |    53 GB    |     128     |

## Results with various caching policies

This part trains a three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 10GB CPU cache, using neighbor sampling.

Below results are collected on an AWS EC2 g5.8xlarge instance with 128GB RAM, 32 cores, an 24GB A10G GPU and a instance storage of 250K IOPS.

### SIEVE

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --dataset=ogbn-papers100M --epochs=3
```

Result:

```
Training: 1178it [04:10,  4.70it/s, num_nodes=665433, gpu_cache_miss=1, cpu_cache_miss=0.0802]
Evaluating: 123it [00:16,  7.45it/s, num_nodes=625848, gpu_cache_miss=1, cpu_cache_miss=0.0775]
Epoch 00, Loss: 1.4154, Approx. Train: 0.5776, Approx. Val: 0.6248, Time: 250.58920621871948s
Training: 1178it [01:56, 10.11it/s, num_nodes=669761, gpu_cache_miss=1, cpu_cache_miss=0.0558]
Evaluating: 123it [00:14,  8.30it/s, num_nodes=625985, gpu_cache_miss=1, cpu_cache_miss=0.0553]
Epoch 01, Loss: 1.1435, Approx. Train: 0.6385, Approx. Val: 0.6523, Time: 116.4649817943573s
Training: 1178it [01:58,  9.93it/s, num_nodes=663480, gpu_cache_miss=1, cpu_cache_miss=0.0478]
Evaluating: 123it [00:14,  8.56it/s, num_nodes=624204, gpu_cache_miss=1, cpu_cache_miss=0.0476]
Epoch 02, Loss: 1.0948, Approx. Train: 0.6516, Approx. Val: 0.6590, Time: 118.57369756698608s
```

### S3-FIFO

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --cpu-feature-cache-policy=s3-fifo --dataset=ogbn-papers100M --epochs=3
```

Result:

```
Training: 1178it [04:23,  4.48it/s, num_nodes=668079, gpu_cache_miss=1, cpu_cache_miss=0.0844]
Evaluating: 123it [00:23,  5.23it/s, num_nodes=623303, gpu_cache_miss=1, cpu_cache_miss=0.0831]
Epoch 00, Loss: 1.4094, Approx. Train: 0.5798, Approx. Val: 0.6184, Time: 263.1500926017761s
Training: 1178it [02:59,  6.55it/s, num_nodes=658380, gpu_cache_miss=1, cpu_cache_miss=0.071] 
Evaluating: 123it [00:22,  5.37it/s, num_nodes=627476, gpu_cache_miss=1, cpu_cache_miss=0.0708]
Epoch 01, Loss: 1.1435, Approx. Train: 0.6385, Approx. Val: 0.6599, Time: 179.91719889640808s
Training: 1178it [02:57,  6.64it/s, num_nodes=661097, gpu_cache_miss=1, cpu_cache_miss=0.0665]
Evaluating: 123it [00:22,  5.40it/s, num_nodes=624593, gpu_cache_miss=1, cpu_cache_miss=0.0665]
Epoch 02, Loss: 1.0952, Approx. Train: 0.6514, Approx. Val: 0.6602, Time: 177.34704399108887s
```

### LRU

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --cpu-feature-cache-policy=lru --dataset=ogbn-papers100M --epochs=3
```

Result:

```
Training: 1178it [04:25,  4.44it/s, num_nodes=660848, gpu_cache_miss=1, cpu_cache_miss=0.0849]
Evaluating: 123it [00:20,  6.03it/s, num_nodes=626265, gpu_cache_miss=1, cpu_cache_miss=0.0828]
Epoch 00, Loss: 1.4170, Approx. Train: 0.5780, Approx. Val: 0.6372, Time: 265.1538300514221s
Training: 1178it [02:37,  7.46it/s, num_nodes=663757, gpu_cache_miss=1, cpu_cache_miss=0.0644]
Evaluating: 123it [00:19,  6.27it/s, num_nodes=627385, gpu_cache_miss=1, cpu_cache_miss=0.0642]
Epoch 01, Loss: 1.1450, Approx. Train: 0.6390, Approx. Val: 0.6355, Time: 157.8178265094757s
Training: 1178it [02:37,  7.46it/s, num_nodes=676522, gpu_cache_miss=1, cpu_cache_miss=0.0581]
Evaluating: 123it [00:19,  6.27it/s, num_nodes=625828, gpu_cache_miss=1, cpu_cache_miss=0.0581]
Epoch 02, Loss: 1.0964, Approx. Train: 0.6518, Approx. Val: 0.6547, Time: 157.93660283088684s
```

### CLOCK

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --cpu-feature-cache-policy=clock --dataset=ogbn-papers100M --epochs=3
```

Result:

```
Training: 1178it [04:52,  4.02it/s, num_nodes=662105, gpu_cache_miss=1, cpu_cache_miss=0.0938]
Evaluating: 123it [00:28,  4.30it/s, num_nodes=625796, gpu_cache_miss=1, cpu_cache_miss=0.0931]
Epoch 00, Loss: 1.4158, Approx. Train: 0.5782, Approx. Val: 0.6375, Time: 292.89667797088623s
Training: 1178it [04:03,  4.83it/s, num_nodes=677917, gpu_cache_miss=1, cpu_cache_miss=0.0859]
Evaluating: 123it [00:29,  4.15it/s, num_nodes=626673, gpu_cache_miss=1, cpu_cache_miss=0.086]
Epoch 01, Loss: 1.1436, Approx. Train: 0.6388, Approx. Val: 0.6482, Time: 243.7826919555664s
Training: 1178it [04:13,  4.65it/s, num_nodes=671817, gpu_cache_miss=1, cpu_cache_miss=0.0845]
Evaluating: 123it [00:30,  4.08it/s, num_nodes=629312, gpu_cache_miss=1, cpu_cache_miss=0.0847]
Epoch 02, Loss: 1.0960, Approx. Train: 0.6512, Approx. Val: 0.6572, Time: 253.2754566669464s
```
