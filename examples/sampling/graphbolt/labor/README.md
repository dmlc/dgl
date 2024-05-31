Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
============

- Paper link: [https://papers.nips.cc/paper_files/paper/2023/hash/51f9036d5e7ae822da8f6d4adda1fb39-Abstract-Conference.html](NeurIPS 2023)
This is an official Labor sampling example to showcase the use of [https://docs.dgl.ai/en/latest/generated/dgl.graphbolt.LayerNeighborSampler.html](dgl.graphbolt.LayerNeighborSampler).

This sampler has 2 parameters, `layer_dependency=[False|True]` and
`batch_dependency=k`, where k is any nonnegative integer.

We use early stopping so that the final accuracy numbers are reported with a
fairly well converged model. Additional contributions to improve the validation
accuracy are welcome, and hence hopefully also improving the test accuracy.

### layer_dependency

Enabling this parameter by the command line option `--layer-dependency` makes it so
that the random variates for sampling are identical across layers. This ensures
that the same vertex gets the same neighborhood in each layer.

### batch_dependency

This method is proposed in Section 3.2 of [https://arxiv.org/pdf/2310.12403](Cooperative Minibatching in Graph Neural Networks), it is denoted as kappa in the paper. It
makes the random variates used across minibatches dependent, thus increasing 
temporal locality. When used with a cache, the increase in the temporal locality
can be observed by monitoring the drop in the cache miss rate with higher values
of the batch dependency parameter, speeding up embedding transfers to the GPU.

### Performance

Use the `--torch-compile` option for best performance. If your GPU has spare
memory, consider using `--mode=cuda-cuda-cuda` to move the whole dataset to the
GPU. If not, consider using `--mode=cuda-pinned-cuda --num-gpu-cached-features=N`
to keep the graph on the GPU and features in system RAM with `N` of the node
features cached on the GPU. If you can not even fit the graph on the GPU, then
consider using `--mode=pinned-pinned-cuda --num-gpu-cached-features=N`. Finally,
you can use `--mode=cpu-pinned=cuda --num-gpu-cached-features=N` to perform the
sampling operation on the CPU.

### Examples

We use `--num-gpu-cached-features=500000` to cache the 500k of the node
embeddings for the `ogbn-products` dataset (default). Check the command line
arguments to see which other datasets can be run. When running with the yelp
dataset, using `--dropout=0` gives better final validation and test accuracy.

Example run with batch_dependency=1, cache miss rate is 62%:

```bash
python node_classification.py --num-gpu-cached-features=500000 --batch-dependency=1
Training in pinned-pinned-cuda mode.
Loading data...
The dataset is already preprocessed.
Training: 192it [00:03, 50.95it/s, num_nodes=247243, cache_miss=0.619]
Evaluating: 39it [00:00, 76.01it/s, num_nodes=137466, cache_miss=0.621]
Epoch 00, Loss: 1.1161, Approx. Train: 0.7024, Approx. Val: 0.8612, Time: 3.7688188552856445s
```

Example run with batch_dependency=32, cache miss rate is 22%:

```bash
python node_classification.py --num-gpu-cached-features=500000 --batch-dependency=32
Training in pinned-pinned-cuda mode.
Loading data...
The dataset is already preprocessed.
Training: 192it [00:03, 54.34it/s, num_nodes=250479, cache_miss=0.221]
Evaluating: 39it [00:00, 84.66it/s, num_nodes=135142, cache_miss=0.226]
Epoch 00, Loss: 1.1288, Approx. Train: 0.6993, Approx. Val: 0.8607, Time: 3.5339605808258057s
```

Example run with layer_dependency=True, # sampled nodes is 190k vs 250k without
this option:

```bash
python node_classification.py --num-gpu-cached-features=500000 --layer-dependency
Training in pinned-pinned-cuda mode.
Loading data...
The dataset is already preprocessed.
Training: 192it [00:03, 54.03it/s, num_nodes=191259, cache_miss=0.626]
Evaluating: 39it [00:00, 79.49it/s, num_nodes=108720, cache_miss=0.627]
Epoch 00, Loss: 1.1495, Approx. Train: 0.6932, Approx. Val: 0.8586, Time: 3.5540308952331543s
```

Example run with the original GraphSAGE sampler (Neighbor Sampler), # sampled nodes 
is 520k, more than 2x higher than Labor sampler.

```bash
python node_classification.py --num-gpu-cached-features=500000 --sample-mode=sample_neighbor
Training in pinned-pinned-cuda mode.
Loading data...
The dataset is already preprocessed.
Training: 192it [00:04, 45.60it/s, num_nodes=517522, cache_miss=0.563]
Evaluating: 39it [00:00, 77.53it/s, num_nodes=255686, cache_miss=0.565]
Epoch 00, Loss: 1.1152, Approx. Train: 0.7015, Approx. Val: 0.8652, Time: 4.211000919342041s
```
