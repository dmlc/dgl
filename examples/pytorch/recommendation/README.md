# Recommendation model

## PinSage

NOTE: this version is not using NodeFlow yet.  Also, this is a slow version.

First, download and extract from https://dgl.ai.s3.us-east-2.amazonaws.com/dataset/ml-1m.tar.gz

One can then run the following to train PinSage on MovieLens-1M:

```bash
python3 main.py --opt Adam --lr 1e-3 --sched none --sgd-switch 25
```

One can also incorporate user and movie features into training:

```bash
python3 main.py --opt Adam --lr 1e-3 --sched none --sgd-switch 25 --use-feature
```

Currently, performance of PinSage on MovieLens-1M has the best mean reciprocal rank of
0.032298±0.048078 on validation (and 0.033695±0.051963 on test set for the same model).
The Implicit Factorization Model from Spotlight has a 0.034572±0.041653 on the test set.

## GraphSage with distributed sampling

GraphSage with distributed sampling supports the following datasets:

* [MovieLens-1M](https://dgl.ai.s3.us-east-2.amazonaws.com/dataset/ml-1m.tar.gz) (`movielens1m`)
* [MovieLens-10M](https://dgl.ai.s3.us-east-2.amazonaws.com/dataset/ml-10m.zip) (`movielens10m`, or `movielens10m-imp` for implicit feedback prediction)
* [MovieLens-20M](https://dgl.ai.s3.us-east-2.amazonaws.com/dataset/ml-20m.zip) (`movielens20m`)
