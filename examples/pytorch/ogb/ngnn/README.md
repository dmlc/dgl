# NGNN + GraphSage/GCN

## Introduction

This is an example of implementing [NGNN](https://arxiv.org/abs/2111.11638) for link prediction in DGL.

We use a model-agnostic methodology, namely Network In Graph Neural Network (NGNN), which allows arbitrary GNN models to increase their model capacity.

The script in this folder experiments full-batch GCN/GraphSage (with/without NGNN) on the datasets: ogbl-ddi, ogbl-collab and ogbl-ppa.

## Installation requirements
```
ogb>=1.3.3
torch>=1.11.0
dgl>=0.8
```

## Experiments

We do not fix random seeds at all, and take over 10 runs for all models. All models are trained on a single V100 GPU with 16GB memory.

### ogbl-ddi

#### performance

<table>
   <tr>
      <th></th>
      <th colspan=3 style="text-align: center;">test set</th>
      <th colspan=3 style="text-align: center;">validation set</th>
      <th>#parameters</th>
   </tr>
   <tr>
      <td></td>
      <td>Hits@20</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td>Hits@20</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td></td>
   </tr>
   <tr>
      <td>GCN+NGNN(paper)</td>
      <td>48.22% ± 7.00%</td>
      <td>82.56% ± 4.03%</td>
      <td>89.48% ± 1.68%</td>
      <td>65.95% ± 1.16%</td>
      <td>70.24% ± 0.50%</td>
      <td>72.54% ± 0.62%</td>
      <td rowspan=2>1,487,361</td>
   </tr>
   <tr>
      <td>GCN+NGNN(ours; 50runs)</td>
      <td><b>54.83% ± 15.81%</b></td>
      <td><b>93.15% ± 2.59%</b></td>
      <td><b>97.05% ± 0.56%</b></td>
      <td>71.21% ± 0.38%</td>
      <td>73.55% ± 0.25%</td>
      <td>76.24% ± 1.33%</td>
   </tr>
   <tr>
      <td>GraphSage+NGNN(paper)</td>
      <td>60.75% ± 4.94%</td>
      <td>84.58% ± 1.89%</td>
      <td>92.58% ± 0.88%</td>
      <td>68.05% ± 0.68%</td>
      <td>71.14% ± 0.33%</td>
      <td>72.77% ± 0.09%</td>
      <td rowspan=2>1,618,433</td>
   </tr>
   <tr>
      <td>GraphSage+NGNN(ours; 50runs)</td>
      <td>57.70% ± 15.23%</td>
      <td><b>96.18% ± 0.94%</b></td>
      <td><b>98.58% ± 0.17%</b></td>
      <td>73.23% ± 0.40%</td>
      <td>87.20% ± 5.29%</td>
      <td>98.71% ± 0.22%</td>
   </tr>
</table>

A 3-layer MLP is used as LinkPredictor here, while a 2-layer one is used by the NGNN paper. This is the main reason for the better performance.

#### Reproduction of performance

- GCN + NGNN
```{.bash}
python main.py --dataset ogbl-ddi --device 0 --ngnn_type input --epochs 800 --dropout 0.5 --num_layers 2 --lr 0.0025 --batch_size 16384 --runs 50
```

- GraphSage + NGNN
```{.bash}
python main.py --dataset ogbl-ddi --device 1 --ngnn_type input --use_sage --epochs 600 --dropout 0.25 --num_layers 2 --lr 0.0012 --batch_size 32768 --runs 50
```

### ogbl-collab

#### Performance

<table>
   <tr>
      <th></th>
      <th colspan=3 style="text-align: center;">test set</th>
      <th colspan=3 style="text-align: center;">validation set</th>
      <th>#parameters</th>
   </tr>
   <tr>
      <td></td>
      <td>Hits@10</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td>Hits@10</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td></td>
   </tr>
   <tr>
      <td>GCN+NGNN(paper)</td>
      <td>36.69% ± 0.82%</td>
      <td>51.83% ± 0.50%</td>
      <td>57.41% ± 0.22%</td>
      <td>44.97% ± 0.97%</td>
      <td>60.84% ± 0.63%</td>
      <td>66.09% ± 0.30%</td>
      <td rowspan=2>428,033</td>
   </tr>
   <tr>
      <td>GCN+NGNN(ours)</td>
      <td><b>39.29% ± 1.21%</b></td>
      <td><b>53.48% ± 0.40%</b></td>
      <td>58.34% ± 0.45%</td>
      <td>48.28% ± 1.39%</td>
      <td>62.73% ± 0.40%</td>
      <td>67.13% ± 0.39%</td>
   </tr>
   <tr>
      <td>GraphSage+NGNN(paper)</td>
      <td>36.83% ± 2.56%</td>
      <td>52.62% ± 1.04%</td>
      <td>57.96% ± 0.56%</td>
      <td>45.62% ± 2.56%</td>
      <td>61.34% ± 1.05%</td>
      <td>66.26% ± 0.44%</td>
      <td rowspan=2>591,873</td>
   </tr>
   <tr>
      <td>GraphSage+NGNN(ours)</td>
      <td><b>40.30% ± 1.03%</b></td>
      <td>53.59% ± 0.56%</td>
      <td>58.75% ± 0.57%</td>
      <td>49.85% ± 1.07%</td>
      <td>62.81% ± 0.46%</td>
      <td>67.33% ± 0.38%</td>
   </tr>
</table>

#### Reproduction of performance

- GCN + NGNN
```{.bash}
python main.py --dataset ogbl-collab --device 2 --ngnn_type hidden --epochs 600 --dropout 0.2 --num_layers 3 --lr 0.001 --batch_size 32768 --runs 10
```

- GraphSage + NGNN
```{.bash}
python main.py --dataset ogbl-collab --device 3 --ngnn_type input --use_sage --epochs 800 --dropout 0.2 --num_layers 3 --lr 0.0005 --batch_size 32768 --runs 10
```

### ogbl-ppa

#### Performance

<table>
   <tr>
      <th></th>
      <th colspan=3 style="text-align: center;">test set</th>
      <th colspan=3 style="text-align: center;">validation set</th>
      <th>#parameters</th>
   </tr>
   <tr>
      <td></td>
      <td>Hits@10</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td>Hits@10</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td></td>
   </tr>
   <tr>
      <td>GCN+NGNN(paper)</td>
      <td>5.64% ± 0.93%</td>
      <td>18.44% ± 1.88%</td>
      <td>26.78% ± 0.9%</td>
      <td>8.14% ± 0.71%</td>
      <td>19.69% ± 0.94%</td>
      <td>27.86% ± 0.81%</td>
      <td rowspan=1>673,281</td>
   </tr>
   <tr>
      <td>GCN+NGNN(ours)</td>
      <td><b>13.07% ± 3.24%</b></td>
      <td><b>28.55% ± 1.62%</b></td>
      <td><b>36.83% ± 0.99%</b></td>
      <td>16.36% ± 1.89%</td>
      <td>30.56% ± 0.72%</td>
      <td>38.34% ± 0.82%</td>
      <td>410,113</td>
   </tr>
   <tr>
      <td>GraphSage+NGNN(paper)</td>
      <td>3.52% ± 1.24%</td>
      <td>15.55% ± 1.92%</td>
      <td>24.45% ± 2.34%</td>
      <td>5.59% ± 0.93%</td>
      <td>17.21% ± 0.69%</td>
      <td>25.42% ± 0.50%</td>
      <td rowspan=1>819,201</td>
   </tr>
   <tr>
      <td>GraphSage+NGNN(ours)</td>
      <td><b>11.73% ± 2.42%</b></td>
      <td><b>29.88% ± 1.84%</b></td>
      <td><b>40.05% ± 1.38%</b></td>
      <td>14.73% ± 2.36%</td>
      <td>31.59% ± 1.72%</td>
      <td>40.58% ± 1.23%</td>
      <td>556,033</td>
   </tr>
</table>

The main difference between this implementation and NGNN paper is the position of NGNN (all -> input).

#### Reproduction of performance

- GCN + NGNN
```{.bash}
python main.py --dataset ogbl-ppa --device 4 --ngnn_type input --epochs 80 --dropout 0.2 --num_layers 3 --lr 0.001 --batch_size 49152 --runs 10
```

- GraphSage + NGNN
```{.bash}
python main.py --dataset ogbl-ppa --device 5 --ngnn_type input --use_sage --epochs 80 --dropout 0.2 --num_layers 3 --lr 0.001 --batch_size 49152 --runs 10
```

## References

```{.tex}
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
```
