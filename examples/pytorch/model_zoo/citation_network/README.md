# Node Classification on Citation Networks

This example shows how to use modules defined in `dgl.nn.pytorch.conv` to do node classification on
citation network datasets.

## Datasets

- Cora
- Citeseer
- Pubmed

## Models

- GCN: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907)
- GAT: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- GraphSAGE [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- APPNP: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/pdf/1810.05997)
- GIN: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)
- TAGCN: [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370)
- SGC: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
- AGNN: [Attention-based Graph Neural Network for Semi-supervised Learning](https://arxiv.org/pdf/1803.03735.pdf)
- ChebNet: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)

## Usage

```
python run.py [--gpu GPU] --model MODEL_NAME --dataset DATASET_NAME [--self-loop]
```

The hyperparameters might not be the optimal, you could specify them manually in `conf.py`.
