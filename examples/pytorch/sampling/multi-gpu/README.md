Multi-GPU training by graph sampling
===

The example trains GNN models on a single machine but multiple GPUs by sampling from a single graph data. The setting is suitable for the case where the graph is too large to fit in one GPU memory.
* The script first load the graph data into CPU memory.
* It then launches multiple identical processes. Each for training on one GPU.
* Each process samples from the input graph using DGL's neighbor sampler.
* Each process then trains on the sample, calculates gradients and performs synchronization on model parameters.

The data set used in this example is the Reddit graph from [Inductive Representation Learning on Large Graphs](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf).

The example includes three models: GCN, GraphSAGE and GAT.

Results
---

Testbed: one AWS EC2 p3.16xlarge instance, 8 V100 GPUs, 64 VCPUs

**GCN:** `python main.py --gpu=0 --model=gcn --num-workers=4`

Accuracy: 93.87%

| | Speed (K samples/sec) |
| --- | --- |
| 1 GPU | 10.8 |
| 2 GPU | 24.1 |
| 4 GPU | 41.5 |
| 8 GPU | 69.1 |

**GraphSAGE:** `python main.py --gpu=0 --model=sage --num-workers=4`

Accuracy: 95.04%

| | Speed (K samples/sec) |
| --- | --- |
| 1 GPU | 10.8 |
| 2 GPU | 23.3 |
| 4 GPU | 40.5 |
| 8 GPU | 59.5 |

**GAT:** `python main.py --gpu=0 --model=gat --num-workers=4 --lr=0.01`

Accuracy: 93.7%

| | Speed (K samples/sec) |
| --- | --- |
| 1 GPU | 5.11 |
| 2 GPU | 10.4 |
| 4 GPU | 19.9 |
| 8 GPU | 33.1 |
