# Model Examples using DGL (w/ Pytorch backend)

Each model is hosted in their own folders. Please read their README.md to see how to
run them.

To understand step-by-step how these models are implemented in DGL. Check out our
[tutorials](https://docs.dgl.ai/tutorials/models/index.html)

## Model summary

Here is a summary of the model accuracy and training speed. Our testbed is Amazon EC2 p3.2x instance (w/ V100 GPU).

| Model | Reported <br> Accuracy | DGL <br> Accuracy | Author's training speed (epoch time) | DGL speed (epoch time) | Improvement |
| ----- | ----------------- | ------------ | ------------------------------------ | ---------------------- | ----------- |
| [GCN](https://arxiv.org/abs/1609.02907)  | 81.5% | 81.0% | [0.0051s (TF)](https://github.com/tkipf/gcn) | 0.0031s | 1.64x |
| [GAT](https://arxiv.org/abs/1710.10903)  | 83.0% | 83.9% | [0.0982s (TF)](https://github.com/PetarV-/GAT) | 0.0113s | 8.69x |
| [SGC](https://arxiv.org/abs/1902.07153) | 81.0% | 81.9% | n/a | 0.0008s | n/a |
| [TreeLSTM](http://arxiv.org/abs/1503.00075) | 51.0% | 51.72% | [14.02s (DyNet)](https://github.com/clab/dynet/tree/master/examples/treelstm) | 3.18s | 4.3x |
| [R-GCN <br> (classification)](https://arxiv.org/abs/1703.06103) | 73.23% | 73.53% | [0.2853s (Theano)](https://github.com/tkipf/relational-gcn) | 0.0075s | 38.2x |
| [R-GCN <br> (link prediction)](https://arxiv.org/abs/1703.06103) | 0.158 | 0.151 | [2.204s (TF)](https://github.com/MichSchli/RelationPrediction) | 0.453s | 4.86x |
| [JTNN](https://arxiv.org/abs/1802.04364) | 96.44% | 96.44% | [1826s (Pytorch)](https://github.com/wengong-jin/icml18-jtnn) | 743s | 2.5x |
| [LGNN](https://arxiv.org/abs/1705.08415) | 94% | 94% | n/a | 1.45s | n/a |
| [DGMG](https://arxiv.org/pdf/1803.03324.pdf) | 84% | 90% | n/a | 238s | n/a |
| [GraphWriter](https://www.aclweb.org/anthology/N19-1238.pdf) | 14.31(BLEU) | 14.3(BLEU) | 1970s | 1192s | 1.65x |
