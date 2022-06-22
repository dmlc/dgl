# Heterogeneous Graph Transformer (HGT)

[Alternative PyTorch-Geometric implementation](https://github.com/acbull/pyHGT)

[“**Heterogeneous Graph Transformer**”](https://arxiv.org/abs/2003.01332) is a graph neural network architecture that can deal with large-scale heterogeneous and dynamic graphs.


This toy experiment is based on DGL's official [tutorial](https://docs.dgl.ai/en/0.4.x/generated/dgl.heterograph.html). As the ACM datasets doesn't have input feature, we simply randomly assign features for each node. Such process can be simply replaced by any prepared features.


The reference performance against R-GCN and MLP running 5 times:


| Model        | Test Accuracy    | # Parameter  |
| ---------    | ---------------  | -------------|
| 2-layer HGT  | 0.465 ± 0.007   |  2,176,324   |
| 2-layer RGCN | 0.392 ± 0.013    |  416,340   |
| MLP          | 0.132 ± 0.003    |  200,974     | 
