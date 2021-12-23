# Inductive Graph-based Matrix Completion(IGMC)

Paper link: [https://arxiv.org/abs/1904.12058](https://arxiv.org/abs/1904.12058)
Author's code: [https://github.com/muhanzhang/IGMC](https://github.com/muhanzhang/IGMC)

Credit: Junfeng Zhou ([@zhoujf620](https://github.com/zhoujf620)), Jiahang Li ([@LspongebobJH](https://github.com/LspongebobJH))

## Dependencies

* PyTorch 1.10.1+
* DGL 0.7.1+

## Datasets

Supported datasets: ml-100k, ml-1m

|              | ml-100k       | ml-1m         |
| ------------ | ------------- | ------------- |
| Users        | 943           | 6,040         |
| Items        | 1,682         | 3,706         |
| Ratings      | 100,000       | 1,000,209     |
| Density      | 0.0630        | 0.0447        |
| Rating types | 1, 2, 3, 4, 5 | 1, 2, 3, 4, 5 |



## How to run

- ml-100k
  - run on one GPU


```shell
python3 train.py --data_name ml-100k --testing
```

- ml-1m
  - distributed training on multiple GPUs


```shell
python3 train_multi_gpu.py --data_name ml-1m --testing --gpu 0,1,2,3
```

## Results

|Dataset|Our code <br> best of epochs|Author code <br> best of epochs / ensembled|
|:-:|:-:|:-:|
|ml-100k|0.9041|0.9053 / 0.9051|
|ml-1m|0.8697|0.8685 / 0.8558|
