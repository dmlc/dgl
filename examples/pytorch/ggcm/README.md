# DGL Implementation of GGCM

This DGL example implements the GGCM method from the paper: [From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited](https://arxiv.org/abs/2309.13599).
The authors' original implementation can be found [here](https://github.com/zhengwang100/ogc_ggcm).


## Example Implementor

This example was implemented by [Sinuo Xu](https://github.com/SinuoXu) when she was an undergraduate at SJTU.


## Dependencies
Python 3.11.5<br>
PyTorch 2.0.1<br>
DGL 1.1.2<br>
scikit-learn 1.3.1<br>


## Dataset
The DGL's built-in Citeseer, Cora and Pubmed datasets, as follows:
| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Citeseer | 3,327 | 9,228 | 3,703 | 6 | 120 | 500 | 1000 |
|Cora	|2,708|	10,556|	1,433|	7	|140|	500|	1000|
|Pubmed|	19,717|	88,651|	500	|3|	60|	500|	1000|


## Usage
Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python train.py --dataset citeseer
python train.py --dataset cora --decline 1.0 --alpha 0.15 --epochs 100 --lr 0.2 --layer_num 16 --negative_rate 20.0 --wd 1e-5 --decline_neg 0.5
python train.py --dataset pubmed --decline 1.0 --alpha 0.1 --epochs 100 --lr 0.2 --layer_num 16 --negative_rate 20.0 --wd 2e-5 --decline_neg 0.5
```

## Performance

|Dataset|citeseer|cora|pubmed|
| :-: | :-: | :-: | :-: |
| GGCM (DGL)|74.1|83.5|80.7|
|GGCM (reported) |74.2|83.6|80.8|
