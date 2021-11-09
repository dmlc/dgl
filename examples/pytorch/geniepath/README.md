# DGL Implementation of the GeniePath Paper

This DGL example implements the GNN model proposed in the paper [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910).

Example implementor
----------------------
This example was implemented by [Kay Liu](https://github.com/kayzliu) during his SDE intern work at the AWS Shanghai AI Lab.

Dependencies
----------------------
- Python 3.7.10
- PyTorch 1.8.1
- dgl 0.7.0
- scikit-learn 0.23.2

Dataset
---------------------------------------
The datasets used for node classification are [Pubmed citation network dataset](https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.PubmedGraphDataset) (tranductive) and [Protein-Protein Interaction dataset](https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.PPIDataset) (inductive).

How to run
--------------------------------
If want to train on Pubmed (transductive), run
```
python pubmed.py
```

If want to use a GPU, run
```
python pubmed.py --gpu 0
```

If want to train GeniePath-Lazy, run
```
python pubmed.py --lazy True
```

If want to train on PPI (inductive), run
```
python ppi.py
```

Performance
-------------------------
Dataset: Pubmed (ACC)
|Method | GeniePath|
| ------ | ----------- |
| Paper  | 78.5%       |
| DGL    | 73.0%       |

Dataset: PPI (micro-F1)
|Method | GeniePath| GeniePath-lazy| GeniePath-lazy-residual|
| ------ | ----------- | ------------- | ------------------ |
| Paper  | 0.9520      | 0.9790        | 0.9850        |
| DGL    | 0.9729      | 0.9802        | 0.9798        |
