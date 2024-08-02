## Overview

This project demonstrates how to use GraphBolt to train and evaluate a GraphSAGE model for node classification task on large graphs, where node features are on-disk and fetched using `DiskBasedFeature`. GraphBolt utilizes various in-house implemented caching policy algorithms such as [SIEVE](https://cachemon.github.io/SIEVE-website/), [S3-FIFO](https://s3fifo.com), LRU and [CLOCK](https://people.csail.mit.edu/saltzer/Multics/MHP-Saltzer-060508/bookcases/M00s/M0104%20074-12%29.PDF) to cache frequently required features and io_uring to fetch cache-missed features from disk. The SIEVE algorithm is the default option.

# Node classification task

This example demonstrates how to run node classification task with **GraphBolt.DiskBasedFeature**. All results are collected on an AWS EC2 g5.8xlarge instance with 128GB RAM, 32 cores, an 24GB A10G GPU and a instance storage of 250K IOPS.

## Run on `ogbn-papers100M` dataset

|     Dataset     | Graph Size | Feature Size | Feature Dim |
| :-------------: | :--------: | :----------: | :---------: |
| Ogbn-papers100M |   13 GB   |    53 GB    |     128     |

## Results with various caching policies

This part trains a three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 10GB CPU cache, using neighbor sampling.

### Run default SIEVE policy

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

### Performance Comparison on four caching polices

Below results demonstrate the epoch time with four different caching policies.

| Policy | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :-----: | :---------: | :---------: | :---------: |
|  SIEVE  |   250.589   |   116.465   |   118.574   |
| S3-FiFO |   263.150   |   179.917   |   177.347   |
|   LRU   |   265.154   |   157.818   |   157.937   |
|  CLOCK  |   292.897   |   243.783   |   253.275   |

## Results with Layer-Neighbor Sampling

This part trains a three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 10GB CPU cache, using Layer-Neighbor Sampling and default SIEVE policy.

### Run default `--batch-dependency=1`

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --dataset=ogbn-papers100M --sample-mode=sample_layer_neighbor --batch-dependency=1 --epochs=3
```

Result:

```
Training: 1178it [02:57,  6.63it/s, num_nodes=464607, gpu_cache_miss=1, cpu_cache_miss=0.0803]                     
Evaluating: 123it [00:15,  7.93it/s, num_nodes=463812, gpu_cache_miss=1, cpu_cache_miss=0.0789]                    
Epoch 00, Loss: 1.4134, Approx. Train: 0.5790, Approx. Val: 0.6277, Time: 177.58959531784058s                      
Training: 1178it [01:34, 12.45it/s, num_nodes=476553, gpu_cache_miss=1, cpu_cache_miss=0.0617]                     
Evaluating: 123it [00:14,  8.45it/s, num_nodes=465260, gpu_cache_miss=1, cpu_cache_miss=0.0616]                    
Epoch 01, Loss: 1.1470, Approx. Train: 0.6383, Approx. Val: 0.6406, Time: 94.6372172832489s                        
Training: 1178it [01:31, 12.84it/s, num_nodes=474932, gpu_cache_miss=1, cpu_cache_miss=0.0554]                     
Evaluating: 123it [00:14,  8.70it/s, num_nodes=464145, gpu_cache_miss=1, cpu_cache_miss=0.0554]                    
Epoch 02, Loss: 1.0990, Approx. Train: 0.6506, Approx. Val: 0.6492, Time: 91.75295209884644s
```

### Performance Comparison on different `--batch-dependency`

| batch-dependency | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :--------------: | :---------: | :---------: | :---------: |
|        1        |   177.590   |   94.637   |   91.753   |
|        64        |   174.810   |   82.209   |   81.020   |
|       4096       |   241.959   |   76.377   |   77.082   |

### Effect of `--layer-dependency`

Below results demonstrate the effect of enabling `--layer-dependency` on epoch time when setting `--batch-dependency=1`.

| layer-dependency | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :--------------: | :---------: | :---------: | :---------: |
|      False      |   177.590   |   94.637   |   91.753   |
|       True       |   163.657   |   86.158   |   83.115   |
