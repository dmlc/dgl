# Graph Convolutional Matrix Completion

Paper link: [https://arxiv.org/abs/1706.02263](https://arxiv.org/abs/1706.02263)
Author's code: [https://github.com/riannevdberg/gc-mc](https://github.com/riannevdberg/gc-mc)

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

## Result


