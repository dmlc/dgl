## Overview

This project demonstrates how to use GraphBolt to train and evaluate a GraphSAGE model for node classification task on large graphs, where node features are on-disk and fetched using `DiskBasedFeature`. GraphBolt utilizes various in-house implemented caching policy algorithms such as [SIEVE](https://cachemon.github.io/SIEVE-website/), [S3-FIFO](https://s3fifo.com), LRU and [CLOCK](https://people.csail.mit.edu/saltzer/Multics/MHP-Saltzer-060508/bookcases/M00s/M0104%20074-12%29.PDF) to cache frequently required features and io_uring to fetch cache-missed features from disk. The SIEVE algorithm is the default option.

# Node classification task

This example demonstrates how to run node classification task with **GraphBolt.DiskBasedFeature**. All results are collected on an AWS EC2 g5.8xlarge instance with 128GB RAM, 32 cores, an 24GB A10G GPU and a instance storage of 250K IOPS.

## Run on `ogbn-papers100M` dataset

|     Dataset     | Graph Size | Feature Size | Feature Dim |
| :-------------: | :--------: | :----------: | :---------: |
| ogbn-papers100M |   13 GB   |    53 GB    |     128     |

## Results with various caching policies

This part trains a three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 10GB CPU cache, using neighbor sampling.

### Run default SIEVE policy

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --dataset=ogbn-papers100M --epochs=3
```

Result:

```
Training: 1178it [03:00,  6.53it/s, num_nodes=671260, gpu_cache_miss=1, cpu_cache_miss=0.0578]                                             
Evaluating: 123it [00:16,  7.47it/s, num_nodes=624816, gpu_cache_miss=1, cpu_cache_miss=0.0569]
Epoch 00, Loss: 1.4173, Approx. Train: 0.5787, Approx. Val: 0.6353, Time: 180.33928060531616s                                              
Training: 1178it [01:39, 11.79it/s, num_nodes=648380, gpu_cache_miss=1, cpu_cache_miss=0.0451]                                             
Evaluating: 123it [00:15,  7.90it/s, num_nodes=625373, gpu_cache_miss=1, cpu_cache_miss=0.0451]
Epoch 01, Loss: 1.1446, Approx. Train: 0.6386, Approx. Val: 0.6382, Time: 99.92613315582275s                                               
Training: 1178it [01:36, 12.15it/s, num_nodes=674194, gpu_cache_miss=1, cpu_cache_miss=0.0408]                                             
Evaluating: 123it [00:15,  8.08it/s, num_nodes=628233, gpu_cache_miss=1, cpu_cache_miss=0.0409]
Epoch 02, Loss: 1.0975, Approx. Train: 0.6507, Approx. Val: 0.6535, Time: 96.95083212852478s
```

### Performance Comparison on four caching polices

Below results demonstrate the epoch time with four different caching policies.

| Policy | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :-----: | :---------: | :---------: | :---------: |
|  SIEVE  |   180.339   |   99.926   |   96.951   |
| S3-FiFO |   181.438   |   110.054   |   108.310   |
|   LRU   |   194.583   |   138.352   |   138.369   |
|  CLOCK  |   188.915   |   129.372   |   129.388   |

## Results with Layer-Neighbor Sampling

This part trains a three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 10GB CPU cache, using Layer-Neighbor Sampling and default SIEVE policy.

### Run default `--batch-dependency=1`

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=0 --cpu-cache-size-in-gigabytes=10 --dataset=ogbn-papers100M --sample-mode=sample_layer_neighbor --batch-dependency=1 --epochs=3
```

Result:

```
Training: 1178it [02:51,  6.88it/s, num_nodes=463495, gpu_cache_miss=1, cpu_cache_miss=0.0774]                                             
Evaluating: 123it [00:15,  7.94it/s, num_nodes=465592, gpu_cache_miss=1, cpu_cache_miss=0.0762]
Epoch 00, Loss: 1.4173, Approx. Train: 0.5774, Approx. Val: 0.6300, Time: 171.11454963684082s                                              
Training: 1178it [01:34, 12.43it/s, num_nodes=474446, gpu_cache_miss=1, cpu_cache_miss=0.0604]                                             
Evaluating: 123it [00:14,  8.45it/s, num_nodes=462042, gpu_cache_miss=1, cpu_cache_miss=0.0603]
Epoch 01, Loss: 1.1463, Approx. Train: 0.6384, Approx. Val: 0.6395, Time: 94.7821741104126s                                                
Training: 1178it [01:31, 12.82it/s, num_nodes=479331, gpu_cache_miss=1, cpu_cache_miss=0.0545]                                             
Evaluating: 123it [00:14,  8.67it/s, num_nodes=463628, gpu_cache_miss=1, cpu_cache_miss=0.0546]
Epoch 02, Loss: 1.1000, Approx. Train: 0.6501, Approx. Val: 0.6516, Time: 91.8746063709259s
```

### Performance Comparison on different `--batch-dependency`

| batch-dependency | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :--------------: | :---------: | :---------: | :---------: |
|        1        |   171.114   |   94.782   |   91.875   |
|        64        |   144.241   |   78.749   |   75.270   |
|       4096       |   92.494   |   56.111   |   57.647   |

### Effect of `--layer-dependency`

Below results demonstrate the effect of enabling `--layer-dependency` on epoch time when setting `--batch-dependency=1`.

| layer-dependency | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :--------------: | :---------: | :---------: | :---------: |
|      False      |   171.114   |   94.782   |   91.875   |
|       True       |   159.625   |   86.209   |   83.171   |

## Compared to In-mem Performance

This part trains a three-layer GraphSAGE model for 3 epochs on `ogbn-papers100M` dataset with 20GB CPU cache and 5GB GPU cache, using neighbor sampling. We compare it to the in-mem performance with 5GB GPU cache. Following result demonstrates that with sufficient cache memory, the performance of DiskBasedFeature is not bottlenecked by the cache itself and comparable with in-memory feature stores. Note that the first epoch of training initiates the cache, thus taking longer time.

Instruction:

```
python node_classification.py --gpu-cache-size-in-gigabytes=5 --cpu-cache-size-in-gigabytes=20 --dataset=ogbn-papers100M --epochs=3
```

Result:

|  Feature Store  | Epoch 1 (s) | Epoch 2 (s) | Epoch 3 (s) |
| :--------------: | :---------: | :---------: | :---------: |
| DiskBasedFeature |   143.761   |   32.018   |   31.889   |
|    In-memory    |   28.861   |   28.330   |   28.305   |
