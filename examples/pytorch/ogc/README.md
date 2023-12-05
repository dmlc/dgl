# Optimized Graph Convolution (OGC)

This DGL example implements the OGC method from the paper: [From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited](https://arxiv.org/abs/2309.13599).
With only one trainable layer, OGC is a very simple but powerful graph convolution method.


## Example Implementor

This example was implemented by [Sinuo Xu](https://github.com/SinuoXu) when she was an undergraduate at SJTU.


## Dependencies

Python     3.11.5
PyTorch    2.0.1 
DGL       1.1.2 
scikit-learn 1.3.1


## Dataset

The DGL's built-in Cora, Pubmed and Citeseer datasets, as follows:

| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Citeseer | 3,327 | 9,228 | 3,703 | 6 | 120 | 500 | 1000 |
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3 | 60 | 500 | 1000 |


## Usage

```bash
python main.py --dataset cora
python main.py --dataset citeseer
python main.py --dataset pubmed
```

## Performance

| Dataset | Cora | Citeseer | Pubmed |
| :-: | :-: | :-: | :-: |
| OGC (DGL) | **86.9(±0.2)** | **77.4(±0.1)** | **83.6(±0.1)** |
| OGC (Reported) | **86.9(±0.0)** | **77.4(±0.0)** | 83.4(±0.0) |
