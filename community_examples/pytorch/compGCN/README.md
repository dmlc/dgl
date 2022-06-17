# DGL Implementation of the CompGCN Paper

This DGL example implements the GNN model proposed in the paper [CompositionGCN](https://arxiv.org/abs/1911.03082). 
The author's codes of implementation is in [here](https://github.com/malllabiisc/CompGCN)

Example implementor
----------------------
This example was implemented by [zhjwy9343](https://github.com/zhjwy9343) and [KounianhuaDu](https://github.com/KounianhuaDu) at the AWS Shanghai AI Lab.

Dependencies
----------------------
- pytorch 1.9.0
- dgl 0.7.1
- numpy 1.20.3
- ordered_set 4.0.2

Dataset
---------------------------------------
The datasets used for link predictions are FB15k-237 constructed from Freebase and WN18RR constructed from WordNet. The statistics are summarized as followings:

**FB15k-237** 

- Nodes: 14541
- Relation types: 237
- Reversed relation types: 237
- Train: 272115
- Valid: 17535
- Test: 20466

**WN18RR** 

- Nodes: 40943
- Relation types: 11
- Reversed relation types: 11
- Train: 86835
- Valid: 3034
- Test: 3134

How to run
--------------------------------
First to get the data, one can run 

```python
sh get_fb15k-237.sh
```
```python
sh get_wn18rr.sh
```

Then for FB15k-237, run

```python
python main.py --score_func conve --opn ccorr --gpu 0 --data FB15k-237
```

For WN18RR, run

```python
python main.py --score_func conve --opn ccorr --gpu 0 --data wn18rr
```


Performance
-------------------------
**Link Prediction Results**

| Dataset |        FB15k-237         |          WN18RR          |
|---------| ------------------------ | ------------------------ |
|  Metric |    Paper   /  ours (dgl) |    Paper   /  ours (dgl) |
|   MRR   |    0.355   /    0.348    |    0.479   /    0.466    |
|   MR    |     197    /     208     |    3533    /     3542    |
| Hit@10  |    0.535   /   0.527     |    0.546   /    0.525    |
|  Hit@3  |    0.390   /    0.380    |    0.494   /    0.476    |
|  Hit@1  |    0.264   /    0.259    |    0.443   /    0.435    |




