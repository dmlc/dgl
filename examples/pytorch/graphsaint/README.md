# GraphSAINT

This DGL example implements the paper: GraphSAINT: Graph Sampling Based Inductive Learning Method.

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Jiahang Li ([@ljh1064126026](https://github.com/ljh1064126026))  Tang Liu ([@lt610](https://github.com/lt610))

## Dependencies

- PyTorch 1.8.1
- NumPy 1.19.2
- Scikit-learn 0.23.2
- DGL 0.8.1

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available in [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [Baidu Wangpan (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)). Dataset summary("m" stands for multi-label binary classification, and "s" for single-label.):
| Dataset | Nodes | Edges | Degree | Feature | Classes |
| :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) |
| Flickr | 89,250 | 899,756 | 10 | 500 | 7(s) |
| Reddit | 232,965 | 11,606,919 | 50 | 602 | 41(s) |
| Yelp | 716,847 | 6,977,410 | 10 | 300 | 100 (m) |
| Amazon | 1,598,960 | 132,169,734 | 83 | 200 | 107 (m) |

Note that the PPI dataset here is different from DGL's built-in variant.

## Config

- The config file is `config.py`, which contains best configs for experiments below.
- Please refer to `sampler.py` to see explanations of some key parameters.

### Parameters

| **val_every**                                                | **num_workers_sampler**                                      | **num_subg_sampler**                                            |
| validate per 'val_every' epochs                              | number of workers (processes) specified for internal dataloader in SAINTSampler, which is to pre-sample subgraphs | the maximal number of pre-sampled subgraphs                  |
| **batch_size_sampler**                                          | **num_workers**                                              |                                                              |                                                              |
| batch size of internal dataloader in SAINTSampler            | number of workers (processes) specified for external dataloader in train_sampling.py, which is to sample subgraphs in training phase |                                                              |                                                              |




## Minibatch training

Run with following:
```bash
python train_sampling.py --task $task
```

- `$task` includes `ppi_n, ppi_e, ppi_rw, flickr_n, flickr_e, flickr_rw, reddit_n, reddit_e, reddit_rw, yelp_n, yelp_e, yelp_rw, amazon_n, amazon_e, amazon_rw`. For example, `ppi_n` represents running experiments on dataset `ppi` with `node sampler`

## Experiments

* Paper: results from the paper
* Running: results from experiments with the authors' code
* DGL: results from experiments with the DGL example. The experiment config comes from `config.py`. You can modify parameters in the `config.py` to see different performance of different setup.

### F1-micro

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 | 0.641±0.000 | 0.782±0.004 |
| Running | 0.9628 | 0.5077 | 0.9622 | 0.6393 | 0.7695 |
| DGL     | 0.9730 | 0.5071 | 0.9645 | 0.6444 | 0.8014 |

#### Random edge sampler

| Method      | PPI         | Flickr      | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 | 0.653±0.003 | 0.807±0.001 |
| Running | 0.9810 | 0.5066 | 0.9656 | 0.6531 | 0.8071 |
| DGL     | 0.9815 | 0.5041 | 0.9653 | 0.6516 | 0.7756 |

#### Random walk sampler
| Method      | PPI         | Flickr      | Reddit      | Yelp        | Amazon      |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 | 0.653±0.003 | 0.815±0.001 |
| Running | 0.9812 | 0.5104 | 0.9648      | 0.6527      | 0.8131      |
| DGL     | 0.9820 | 0.5110 | 0.9572      | 0.6508      | 0.8157   |
