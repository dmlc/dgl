# GraphSage/GCN (FULL)

## Introduction

This is an example of implementing [NGNN](https://arxiv.org/abs/2111.11638) for link prediction in DGL.

We use a model-agnostic methodology, namely Network In Graph Neural Network (NGNN), which allows arbitrary GNN models to increase their model capacity.

The script in this folder experiments full GCN/GraphSage (with/without NGNN) on the dataset:ogb-ppa.

## Installation requirements
```
ogb>=1.3.3
torch>=1.11.0
dgl>=0.8
```

## Usage

run `run_ppa.sh`, and you can modify the arguments of the script.

## Experiments

We do not fix random seeds at all, and take over 10 runs for all models.  
All models are trained on a single V100 GPU with 16GB memory.

### Performance

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
      <td>GCN(ogb)</td>
      <td>\</td>
      <td>\</td>
      <td>18.67% ± 1.32%</td>
      <td>\</td>
      <td>\</td>
      <td>18.45% ± 1.40%</td>
      <td rowspan=2>278,529</td>
   </tr>
   <tr>
      <td>GCN(paper)</td>
      <td>4.00% ± 1.46%</td>
      <td>14.23% ± 1.81%</td>
      <td>20.21% ± 1.9%</td>
      <td>5.12% ± 0.56%</td>
      <td>14.37% ± 1.05%</td>
      <td>20.92% ± 1.01%</td>
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
      <td>13.07% ± 3.24%</td>
      <td>28.55% ± 1.62%</td>
      <td>36.83% ± 0.99%</td>
      <td>16.36% ± 1.89%</td>
      <td>30.56% ± 0.72%</td>
      <td>38.34% ± 0.82%</td>
      <td>410,113</td>
   </tr>
   <tr>
      <td>GraphSage(ogb)</td>
      <td>\</td>
      <td>\</td>
      <td>6.55% ± 2.40%</td>
      <td>\</td>
      <td>\</td>
      <td>7.24% ± 2.64%</td>
      <td rowspan=2>424,449</td>
   </tr>
   <tr>
      <td>GraphSage(paper)</td>
      <td>3.68% ± 1.02%</td>
      <td>15.02% ± 1.69%</td>
      <td>23.56% ± 1.58%</td>
      <td>4.94% ± 0.54%</td>
      <td>16.15% ± 1.14%</td>
      <td>23.43% ± 1.39%</td>
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
      <td>11.73% ± 2.42%</td>
      <td>29.88% ± 1.84%</td>
      <td>40.05% ± 1.38%</td>
      <td>14.73% ± 2.36%</td>
      <td>31.59% ± 1.72%</td>
      <td>40.58% ± 1.23%</td>
      <td>556,033</td>
   </tr>
</table>

### Reproduction of performance on OGBL

- GCN + NGNN
```{.bash}
python main.py --device 4 --ngnn_type input --epochs 80 --dropout 0.2 --num_layers 3 --lr 0.001 --batch_size 49152 --runs 10
```

- GraphSage + NGNN
```{.bash}
python main.py --device 5 --ngnn_type input --use_sage --epochs 80 --dropout 0.2 --num_layers 3 --lr 0.001 --batch_size 49152 --runs 10
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
