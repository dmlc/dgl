# Inductive Graph-based Matrix Completion(IGMC)

Paper link: [https://arxiv.org/abs/1904.12058](https://arxiv.org/abs/1904.12058)

Author's code: [https://github.com/muhanzhang/IGMC](https://github.com/muhanzhang/IGMC)

Credit: Junfeng Zhou ([@zhoujf620](https://github.com/zhoujf620)), Jiahang Li ([@LspongebobJH](https://github.com/LspongebobJH))

## Dependencies

* PyTorch 1.13.1+
* DGL 1.0.0+

## Datasets

Supported datasets: ml-100k, ml-1m

|              | ml-100k       | 
| ------------ | ------------- |
| Users        | 943           |
| Items        | 1,682         |
| Ratings      | 100,000       |
| Density      | 0.0630        |
| Rating types | 1, 2, 3, 4, 5 |



## How to run

- ml-100k
```shell
python3 train.py
```

## Results
We report the best RMSE test results among all epochs.
|Dataset|Our code|Author code|
|:-:|:-:|:-:|
<!-- |ml-100k|0.9041|0.9053| -->
|ml-100k|0.9204|0.9053|
