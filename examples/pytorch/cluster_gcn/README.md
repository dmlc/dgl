Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
============
- Paper link: [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953)
- Author's code repo: [https://github.com/google-research/google-research/blob/master/cluster_gcn/](https://github.com/google-research/google-research/blob/master/cluster_gcn/).

This repo reproduce the reported speed and performance maximally on Reddit and PPI. However, the diag enhancement is not covered, as the GraphSage aggregator already achieves satisfying F1 score.

Dependencies
------------
- Python 3.7+(for string formatting features)
- PyTorch 1.9.0+
- scikit-learn
- TorchMetrics 0.11.4

## Run Experiments

```bash
python cluster_gcn.py
```
