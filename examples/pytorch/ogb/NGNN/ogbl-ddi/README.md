# GraphSage/GCN (FULL)

## Introduction

This is an example of implementing [NGNN](https://arxiv.org/abs/2111.11638) for link prediction in DGL.

We use a model-agnostic methodology, namely Network In Graph Neural Network (NGNN), which allows arbitrary GNN models to increase their model capacity.

The script in this folder experiments full GCN/GraphSage (with/without NGNN) on the dataset:ogb-ddi.

## Installation requirements
```
ogb>=1.3.3
torch>=1.11.0
dgl>=0.8
```

## Usage

run `run_ddi.sh`, and you can modify the arguments of the script.

## Experiments

We do not fix random seeds at all, and take over 10 runs for all models. All models are trained on a single V100 GPU with 16GB memory.

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
      <td>Hits@20</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td>Hits@20</td>
      <td>Hits@50</td>
      <td>Hits@100</td>
      <td></td>
   </tr>
   <tr>
      <td>GCN(ogb)</td>
      <td>37.07% ± 5.07%</td>
      <td>\</td>
      <td>\</td>
      <td>55.50% ± 2.08%</td>
      <td>\</td>
      <td>\</td>
      <td rowspan=1>1,289,985</td>
   </tr>
   <tr>
      <td>GCN(paper)</td>
      <td>47.82% ± 5.90%</td>
      <td>79.56% ± 3.83%</td>
      <td>87.58% ± 1.33%</td>
      <td>63.19% ± 0.97%</td>
      <td>68.31% ± 0.18%</td>
      <td>70.89% ± 0.12%</td>
      <td>1,355,777</td>
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
      <td>54.83% ± 15.81%</td>
      <td>93.15% ± 2.59%</td>
      <td>97.05% ± 0.56%</td>
      <td>71.21% ± 0.38%</td>
      <td>73.55% ± 0.25%</td>
      <td>76.24% ± 1.33%</td>
   </tr>
   <tr>
      <td>GraphSage(ogb)</td>
      <td>53.90% ± 4.74%</td>
      <td>\</td>
      <td>\</td>
      <td>62.62% ± 0.37%</td>
      <td>\</td>
      <td>\</td>
      <td rowspan=1>1,421,057</td>
   </tr>
   <tr>
      <td>GraphSage(paper)</td>
      <td>54.27% ± 9.86%</td>
      <td>82.18% ± 4.00%</td>
      <td>91.94% ± 0.64%</td>
      <td>67.54% ± 0.75%</td>
      <td>71.07% ± 0.22%</td>
      <td>72.82% ± 0.23%</td>
      <td>1,486,849</td>
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
      <td>96.18% ± 0.94%</td>
      <td>98.58% ± 0.17%</td>
      <td>73.23% ± 0.40%</td>
      <td>87.20% ± 5.29%</td>
      <td>98.71% ± 0.22%</td>
   </tr>
</table>

### Reproduction of performance on OGBL

- GCN + NGNN
```{.bash}
python main.py --device 2 --ngnn_type input --epochs 800 --dropout 0.5 --num_layers 2 --lr 0.0025 --batch_size 16384 --runs 50
```

- GraphSage + NGNN
```{.bash}
python main.py --device 3 --ngnn_type input --use_sage --epochs 600 --dropout 0.25 --num_layers 2 --lr 0.0012 --batch_size 32768 --runs 50
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
