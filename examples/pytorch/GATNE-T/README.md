Representation Learning for Attributed Multiplex Heterogeneous Network (GANTE)
============

- Paper link: [https://arxiv.org/abs/1905.01669](https://arxiv.org/abs/1905.01669)
- Author's code repo: [https://github.com/THUDM/GATNE](https://github.com/THUDM/GATNE). Note that only GATNE-T is implemented here.

Requirements
------------
- requirements

```bash
pip install -r requirements.txt
```

Also requires PyTorch 1.7.0+.

Datasets
--------

To prepare the datasets:
1. ```bash
   mkdir data
   cd data
   ```
2. Download datasets from the following links:
    - example: https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/example.zip
    - amazon: https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/amazon.zip
    - youtube: https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/youtube.zip
    - twitter: https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/twitter.zip
3. Unzip the datasets

Training
--------

Run with following (available dataset: "example", "youtube", "amazon")
```bash
python src/main.py --input data/example
```

To run on "twitter" dataset, use
```bash
python src/main.py --input data/twitter --eval-type 1 --gpu 0
```

For a big dataset, use sparse to avoid cuda out of memory in backward
```bash
python src/main_sparse.py --input data/example --gpu 0
```

If you have multiple GPUs, you can also accelerate training with [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
```bash
python src/main_sparse_multi_gpus.py --input data/example --gpu 0,1
```

**It is worth noting that DistributedDataParallel will cause more cuda memory consumption and a certain loss of preformance.**


Results
-------
All the results match the [official code](https://github.com/THUDM/GATNE/blob/master/src/main_pytorch.py) with the same hyper parameter values, including twiiter dataset (auc, pr, f1 is 76.29, 76.17, 69.34, respectively).

|         | auc   | pr    | f1    |
| ------- | ----- | ----- | ----- |
| amazon  | 96.88 | 96.31 | 92.12 |
| youtube | 82.29 | 80.35 | 74.63 |
| twitter | 72.40 | 74.40 | 65.89 |
| example | 94.65 | 94.57 | 89.99 |
