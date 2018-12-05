Graph-Based Recommenders Inspired by AutoRec and Graph Convolutional Networks
============

Paper link: [AutoRec](http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf) [CF-NADE](https://arxiv.org/pdf/1605.09477.pdf) [Graph Convolusional Matrix Completion](https://arxiv.org/pdf/1706.02263.pdf)
Author's code repo: [https://github.com/Ian09/CF-NADE](https://github.com/Ian09/CF-NADE)

There are two types of models: (1) AutoRec-inspred where each message is a simple vector embedding of dimension h and (2) CF-NADE-inspired where each message is a matrix of dimension h * num_rating_classes. The latter one has clearer intuitions but more model complexity.

We made the modification to directly use l2-losses instead of the softmax/ordinal losses in some of the referenced papers. This simplifies the implementations. The softmax/ordinal losses, on the other hand, are very specific to rating predictions.

Results
-------

* n-layers=0 realizes a matrix factorization approach
```
# final RMSE: train=0.796, val=0.923, test=0.929.
python cf_autorec.py --dataset 'ml-100k' --n-layers 0
```

* n-layer=1 realizes a GCN approach similar to i-AutoRec.
```
# final RMSE: train=0.907, val=0.945, test=0.952
python cf_autorec.py --dataset 'ml-100k' --n-layers 1 --activation relu
```

* n-layers=0 realizes a tensor decomposition approach
```
# final RMSE: train=0.882, val=0.968, test=0.979
cf_nade.py --dataset 'ml-100k' --n-layers 0 --n-hidden 6 --share
```

* n-layers=1 realizes a GCN tensor decomposition approach
```
# final RMSE: train=0.870, val=0.949, test=0.958
cf_nade.py --dataset 'ml-100k' --n-layers 1 --activation 'relu' --share --n-hidden 6
```


Discussions
-------
On ml-100k, the current results all outperformed average rating prediction baseline, which has RMSE 0.99. The best result came from autorec-inspired matrix factorization, with close margins to the SOTA (0.909). Unfortunately, GCN did not improve the autorec-inspired models, because GCN took more iterations and it was unclear whether it eventually converged or what would be a fair comparison.

On CF-NADE models, GCN had clearer advantages over simple tensor decomposition approaches. This is because the messages are sent through different channels decided by the rating indicators. However, neither tensor decomposition nor GCN achieved state-of-art. There is clear signs of overfitting even with n-hidden=6.

Future work
-------
Extend the experiments on ml-1m and ml-10m.