# Graph Convolutional Matrix Completion

Paper link: [https://arxiv.org/abs/1706.02263](https://arxiv.org/abs/1706.02263)
Author's code: [https://github.com/riannevdberg/gc-mc](https://github.com/riannevdberg/gc-mc)

The implementation does not include mini-batching and thus achieves slightly worse performance
than the reported numbers. Also, ml-10m runs OOM on GPU.

Credit: Jiani Zhang ([@jennyzhang0215](https://github.com/jennyzhang0215))

## Dependencies
* MXNet 1.5.0+
* pandas
* gluonnlp

## Data

Download the MovieLens datasets and unzip into `data_set/ml-?` directory.

```bash
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data_set
```

Supported datasets: ml-100k, ml-1m, ml-10m

## How to run

```bash
DGLBACKEND=mxnet python train.py --data_name=ml-100k
```

## Results

ML-100k: RMSE=0.9204 (0.905 reported)
Ml-1m: RMSE=0.8566 (0.832 reported)
