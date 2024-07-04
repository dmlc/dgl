Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
============

- Paper link: [https://arxiv.org/abs/2210.13339](https://arxiv.org/abs/2210.13339)
This is the official Labor sampling example to reproduce the results in the original
paper with the GraphSAGE GNN model. The model can be changed to any other model where
NeighborSampler can be used.

A more modern and performant version is provided in the
`examples/graphbolt/pyg/labor` folder.

Requirements
------------

```bash
pip install requests lightning==2.0.6 ogb
```

How to run
-------

### Minibatch training for node classification

Train w/ mini-batch sampling on the GPU for node classification on "ogbn-products"

```bash
python3 train_lightning.py --dataset=ogbn-products
```

Results:
```
Test Accuracy: 0.797
```

Any integer passed as the `--importance-sampling=i` argument runs the corresponding
LABOR-i variant. `--importance-sampling=-1` runs the LABOR-* variant.

`--vertex-limit` argument is used if a vertex sampling budget is needed. It adjusts
the batch size at the end of every epoch so that the average number of sampled vertices
converges to the provided vertex limit. Can be used to replicate the vertex sampling
budget experiments in the Labor paper.

During training runs, statistics about number of sampled vertices, edges,
cache miss rates will be reported. One can use tensorboard to look at their plots
during/after training:

```bash
tensorboard --logdir tb_logs
```

## Utilize a GPU feature cache for UVA training

```bash
python3 train_lightning.py --dataset=ogbn-products --use-uva --cache-size=500000
```

## Reduce GPU feature cache miss rate for UVA training

```bash
python3 train_lightning.py --dataset=ogbn-products --use-uva --cache-size=500000 --batch-dependency=64
```

## Force all layers to share the same neighborhood for shared vertices

```bash
python3 train_lightning.py --dataset=ogbn-products --layer-dependency
```