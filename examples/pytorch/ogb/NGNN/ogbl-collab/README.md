# GraphSage/GCN (FULL)

## Introduction

This is an example of implementing [NGNN](https://arxiv.org/abs/2111.11638) for link prediction in DGL.

We use a model-agnostic methodology, namely Network In Graph Neural Network (NGNN), which allows arbitrary GNN models to increase their model capacity.

The script in this folder experiments full GCN/GraphSage (with/without NGNN) on the dataset:ogb-collab.

## Installation requirements
```
ogb>=1.3.3
torch>=1.11.0
dgl>=0.8
```

## Usage

run `run_collab.sh`, and you can modify the arguments of the script.

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
      <td>44.75% ± 1.07%</td>
      <td>\</td>
      <td>\</td>
      <td>52.63% ± 1.15%</td>
      <td>\</td>
      <td rowspan=2>296,449</td>
   </tr>
   <tr>
      <td>GCN(paper)</td>
      <td>35.94% ± 1.60%</td>
      <td>49.52% ± 0.70%</td>
      <td>55.74% ± 0.44%</td>
      <td>43.64% ± 1.97%</td>
      <td>57.90% ± 0.57%</td>
      <td>63.93% ± 0.33%</td>
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
      <td>GraphSage(ogb)</td>
      <td>\</td>
      <td>48.10% ± 0.81%</td>
      <td>\</td>
      <td>\</td>
      <td>56.88% ± 0.77%</td>
      <td>\</td>
      <td rowspan=2>460,289</td>
   </tr>
   <tr>
      <td>GraphSage(paper)</td>
      <td>32.59% ± 3.56%</td>
      <td>51.66% ± 0.35%</td>
      <td>56.91% ± 0.72%</td>
      <td>41.36% ± 3.88%</td>
      <td>60.52% ± 0.52%</td>
      <td>65.55% ± 0.67%</td>
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

### Reproduction of performance on OGBL

- GCN + NGNN
```{.bash}
python main.py --device 0 --ngnn_type hidden --epochs 600 --dropout 0.2 --num_layers 3 --lr 0.001 --batch_size 32768 --runs 10
```

- GraphSage + NGNN
```{.bash}
python main.py --device 1 --ngnn_type input --use_sage --epochs 800 --dropout 0.2 --num_layers 3 --lr 0.0005 --batch_size 32768 --runs 10
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
