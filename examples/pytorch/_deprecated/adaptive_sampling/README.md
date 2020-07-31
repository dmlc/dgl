# Adaptive sampling for graph representation learning

This is dgl implementation of [Adaptive Sampling Towards Fast Graph Representation Learning](https://arxiv.org/abs/1809.05343).

The authors' implementation can be found [here](https://github.com/huangwb/AS-GCNN).

## Performance

Test accuracy on cora dataset achieves 0.84 around 250 epochs when sample size is set to 256 for each layer.

## Usage

`python adaptive_sampling.py --batch_size 20  --node_per_layer 40`