# GraphSAINT

This DGL example implements the paper: GraphSAINT: Graph Sampling Based Inductive Learning Method.

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Jiahang Li ([@ljh1064126026](https://github.com/ljh1064126026))  Tang Liu ([@lt610](https://github.com/lt610))

## Dependencies

- Python 3.7.0
- PyTorch 1.6.0
- NumPy 1.19.2
- Scikit-learn 0.23.2
- DGL 0.5.3

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available in [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [Baidu Wangpan (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)). Once you download the datasets, you need to rename graphsaintdata to data. Dataset summary("m" stands for multi-label classification, and "s" for single-label.):
| Dataset | Nodes | Edges | Degree | Feature | Classes | Train/Val/Test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) | 0.66/0.12/0.22 |
| Flickr | 89,250 | 899,756 | 10 | 500 | 7(s) | 0.50/0.25/0.25 |

Note that the PPI dataset here is different from DGL's built-in variant.

## Config

- The config file is `config.py`.
- Please refer to `sampler.py` to see explanations of some key parameters.

## Minibatch training

Run with following:
```bash
python train_sampling.py --task $task
# e.g. python train_sampling.py --task ppi_n
```

- `$task` includes `ppi_n, ppi_e, ppi_rw, flickr_n, flickr_e, flickr_rw, reddit_n, reddit_e, reddit_rw, yelp_n, yelp_e, yelp_rw, amazon_n, amazon_e, amazon_rw`. For example, `ppi_n` represents running experiments on dataset `ppi` with `node sampler`

## Comparison

* Paper: results from the paper
* Running: results from experiments with the authors' code
* DGL: results from experiments with the DGL example. The experiment config comes from `config.py`. You can modify parameters in `config.py` to see different performance of different setup. 

> Note that we implement offline sampling and online sampling in training phase. Offline sampling means all subgraphs utilized in training phase come from pre-sampled subgraphs. Online sampling means we deprecate all pre-sampled subgraphs and re-sample new subgraphs in training phase.

> Note that the sampling method in pre-sampling phase must be offline sampling.

### F1-micro

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 | 0.641±0.000 | 0.782±0.004 |
| Running | 0.9628 | 0.5077 | 0.9622 | 0.6393 | 0.7695 |
| DGL_offline | 0.9715      | 0.4445      | 0.9645 | 0.6457 | OOM |
| DGL_online | 0.9730 | 0.4358 | 0.9645 | 0.6444 | OOM |

#### Random edge sampler

| Method      | PPI         | Flickr      | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 | 0.653±0.003 | 0.807±0.001 |
| Running | 0.9810 | 0.5066 | 0.9656 | 0.6531 | 0.8071 |
| DGL_offline | 0.9817      | 0.4456      | 0.9655 | 0.6530 | OOM |
| DGL_online | 0.9815 | 0.4508 | 0.9653 | 0.6516 | OOM |

#### Random walk sampler
| Method      | PPI         | Flickr      | Reddit      | Yelp        | Amazon      |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 | 0.653±0.003 | 0.815±0.001 |
| Running | 0.9812 | 0.5104 | 0.9648      | 0.6527      | 0.8131      |
| DGL_offline | 0.9833      | 0.4592      | 0.9582      | 0.6514      | OOM         |
| DGL_online | 0.9820 | 0.4606 | 0.9572      | 0.6508      | OOM         |

### Sampling time

- Here sampling time includes consumed time of pre-sampling subgraphs and calculating normalization coefficients in the beginning.

#### Random node sampler

| Method      | PPI  | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.46 | 3.49 | 19 | 59.01 | 978.62 |
| DGL_offline | 2.51 | 1.12 | 27.32 | 60.15 | OOM |
| DGL_online | 3.17 | 1.11 | 27.82 | 59.42 | OOM |

#### Random edge sampler

| Method      | PPI  | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.4 | 3.18 | 13.88 | 39.02 | 164.35 |
| DGL_offline | 3.08 | 1.91 | 53.59 | 48.94 | OOM |
| DGL_online | 3.04 | 1.87 | 52.01 | 48.38 | OOM |

#### Random walk sampler

| Method      | PPI  | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.7 | 3.82 | 16.97 | 43.25 | 355.68 |
| DGL_offline | 3.10 | 2.18 | 10.57 | 21.52 | OOM |
| DGL_online | 3.05 | 2.13 | 11.01 | 22.23 | OOM |

## Test std of sampling time

- We've run experiments 10 times repeatedly to test average and standard deviation of sampling time. Here we just test sampling time without model performance. Moreover, for efficient testing, the hardware and config employed here are not the same as the experiments above, so the sampling time might be a bit different from the one above. But we keep the environment consistent in all experiments below.

> The value is (average, std)

### Random node sampler

| Method                    | PPI            | Flickr       | Reddit        | Yelp          | Amazon          |
| ------------------------- | -------------- | ------------ | ------------- | ------------- | --------------- |
| DGL_Sampling(std)         | 2.618, 0.004   | 3.017, 0.507 | 35.356, 2.363 | 69.913, 6.3   | 1105.617, 102.1 |
| DGL_Normalization(std)    | 太小可忽略不计 | 0.008, 0.004 | 0.26, 0.047   | 0.189, 0.0288 | 3.025, 0.096    |
|                           |                |              |               |               |                 |
| author_Sampling(std)      | 0.788, 0.661   | 0.728, 0.367 | 8.931, 3.155  | 27.818, 1.384 | bug             |
| author_Normalization(std) | 0.665, 0.565   | 4.981, 2.952 | 17.231, 7.116 | 47.449, 2.794 | bug             |

### Random edge sampler

| Method                    | PPI            | Flickr       | Reddit        | Yelp          | Amazon         |
| ------------------------- | -------------- | ------------ | ------------- | ------------- | -------------- |
| DGL_Sampling(std)         | 3.554, 0.292   | 4.722, 0.245 | 47.09, 2.76   | 75.219, 6.442 | bug            |
| DGL_Normalization(std)    | 太小可忽略不计 | 0.005, 0.007 | 0.235, 0.026  | 0.193, 0.021  | bug            |
|                           |                |              |               |               |                |
| author_Sampling(std)      | 0.802, 0.667   | 0.761, 0.387 | 6.058, 2.166  | 13.914, 1.864 | 90.47, 30.164  |
| author_Normalization(std) | 0.667, 0.570   | 5.180, 3.006 | 15.803, 5.867 | 44.278, 5.853 | 85.309, 28.534 |

### Random walk sampler

| Method                    | PPI          | Flickr       | Reddit        | Yelp          | Amazon          |
| ------------------------- | ------------ | ------------ | ------------- | ------------- | --------------- |
| DGL_Sampling(std)         | 3.304, 0.08  | 5.487, 1.294 | 37.041, 2.083 | 39.951, 3.094 | 179.613, 18.881 |
| DGL_Normalization(std)    | Ignore       | 0.001, 0.003 | 0.235, 0.026  | 0.185, 0.018  | 3.769, 0.326    |
|                           |              |              |               |               |                 |
| author_Sampling(std)      | 0.924, 0.773 | 1.405, 0.718 | 8.608, 3.093  | 19.113, 1.700 | 217.184, 1.546  |
| author_Normalization(std) | 0.701, 0.596 | 5.025, 2.954 | 18.198, 7.223 | 45.874, 8.020 | 128.272, 3.170  |

