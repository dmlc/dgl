# NGNN + SEAL

## Introduction

This is a submission of implementing [NGNN](https://arxiv.org/abs/2111.11638) + [SEAL](https://arxiv.org/pdf/2010.16103.pdf) to OGB link prediction leaderboards. Some code is migrated from [https://github.com/facebookresearch/SEAL_OGB](https://github.com/facebookresearch/SEAL_OGB).

## Installation Requirements
```
ogb>=1.3.4
torch>=1.12.0
dgl>=0.8
scipy, numpy, tqdm...
```

## Experiments

We do not fix random seeds at all, and take over 10 runs for all models. All models are trained on a single T4 GPU with 16GB memory and 96 vCPUs.

### ogbl-ppa

#### performance

|              | Test Hits@100 | Validation Hits@100 | #Parameters |
|:------------:|:-------------------:|:-----------------:|:------------:|
| SEAL | 48.80% ± 3.16% | 51.25% ± 2.52% | 709,122 |
| SEAL + NGNN | 59.71% ± 2.45% | 59.95% ± 2.05% | 735,426 |

#### Reproduction of performance

```{.bash}
python main.py --dataset ogbl-ppa --ngnn_type input --hidden_channels 48 --epochs 50 --lr 0.00015 --batch_size 128 --num_workers 48  --train_percent 5 --val_percent 8 --eval_hits_K 10 --use_feature --dynamic_train --dynamic_val --dynamic_test --runs 10
```

As training is very costly, we select the best model by evaluation on a subset of the validation edges and using a lower K for Hits@K. Then we do experiments on the full validation and test sets with the best model selected, and get the required metrics.  

### ogbl-citation2

#### performance

|              | Test MRR | Validation MRR | #Parameters |
|:------------:|:-------------------:|:-----------------:|:------------:|
| SEAL | 0.8767 ± 0.0032 | 0.8757 ± 0.0031 | 260,802 |
| SEAL + NGNN | 0.8891 ± 0.0022 | 0.8879 ± 0.0022 | 1,134,402 |

#### Reproduction of performance

```{.bash}
python main.py --dataset ogbl-citation2 --ngnn_type all --hidden_channels 256 --epochs 15 --lr 2e-05 --batch_size 64 --num_workers 24  --train_percent 8 --val_percent 4 --num_ngnn_layers 2 --use_feature --use_edge_weight --dynamic_train --dynamic_val --dynamic_test --runs 10
```

For all datasets, if you specify `--dynamic_train`, the enclosing subgraphs of the training links will be extracted on the fly instead of preprocessing and saving to disk. Similarly for `--dynamic_val` and `--dynamic_test`. You can increase `--num_workers` to accelerate the dynamic subgraph extraction process.  
You can also specify the `val_percent` and `eval_hits_K` arguments in the above command to adjust the proportion of the validation dataset to use and the K to use for Hits@K.

## Reference

    @article{DBLP:journals/corr/abs-2111-11638,
      author    = {Xiang Song and
                   Runjie Ma and
                   Jiahang Li and
                   Muhan Zhang and
                   David Paul Wipf},
      title     = {Network In Graph Neural Network},
      journal   = {CoRR},
      volume    = {abs/2111.11638},
      year      = {2021},
      url       = {https://arxiv.org/abs/2111.11638},
      eprinttype = {arXiv},
      eprint    = {2111.11638},
      timestamp = {Fri, 26 Nov 2021 13:48:43 +0100},
      biburl    = {https://dblp.org/rec/journals/corr/abs-2111-11638.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
    @article{zhang2021labeling,
        title={Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning},
        author={Zhang, Muhan and Li, Pan and Xia, Yinglong and Wang, Kai and Jin, Long},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        year={2021}
        }
    
    @inproceedings{zhang2018link,
      title={Link prediction based on graph neural networks},
      author={Zhang, Muhan and Chen, Yixin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5165--5175},
      year={2018}
    }