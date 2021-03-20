# DGL Implementation of the CompGCN Paper

This DGL example implements the GNN model proposed in the paper [CompositionGCN](https://arxiv.org/abs/1911.03082). 
The author's codes of implementation is in [here](https://github.com/malllabiisc/CompGCN)

Example implementor
----------------------
This example was implemented by [zhjwy9343](https://github.com/zhjwy9343) and [KounianhuaDu](https://github.com/KounianhuaDu) at the AWS Shanghai AI Lab.

Dependencies
----------------------
- pytorch 1.4.0
- dgl 0.5.3
- numpy 1.19.4
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

to get the data, or refer to [here](https://github.com/malllabiisc/CompGCN) to download the data.

Then for FB15k-237, run

```python
python main.py -score_func conve -opn ccorr -gpu 0 -data FB15k-237
```

For WN18RR, run

```python
python main.py -score_func conve -opn ccorr -gpu 0 -data WN18RR
```


Performance
-------------------------
**Link Prediction Results**

| Dataset |        FB15k-237         |          WN18RR          |
|---------| ------------------------ | ------------------------ |
|  Metric |    Paper   /  ours (dgl) |    Paper   /  ours (dgl) |
|   MRR   |    0.355   /    0.349    |    0.479   /    0.471    |
|   MR    |     197    /     208     |    3533    /     3550    |
| Hit@10  |    0.535   /   0.526     |    0.546   /    0.532    |
|  Hit@3  |    0.390   /    0.381    |    0.494   /    0.480    |
|  Hit@1  |    0.264   /    0.260    |    0.443   /    0.438    |

To Be Removed Before Merge
-------------------------
**The differences between this model and the MVPExample model**

- Single relation ->  Multi-relation
- Add in the basis vector
- Add in the in/out edge norm
- Dropouts in and after the compGCN layers (Following the paper, in the layer, the droput only acts on the in/out features but dose not act on the loop feature)
- Batchnorm (Following the paper, the batchnorm only acts on the node features but dose not act on the relation features)
- Activation: tanh (This only acts on node features too)
- ConvE link prediction module

**About the node classification**
For the node classification task, the default settings about this task are missing in the paper. The public codes only include the link prediction part without any mention to the node classification part. And in the paper, the main experiment, ablation study, and hyperparameters description all seem to focus on link prediction. The other two tasks (node classification and graph classification) are poorly displayed. Implementing node classification with the current model and settings will get a far worse results than the paper except that we regard the validation and test sets as the same (even doing so, the results can not perfectly match the paper). The paper said "we use 10% training data as validation for selecting the best model for both the datasets". Therefore we cannot use validation results to reach the paper results. Since that we cannot dig into the node classification settings and that the main experiments focus on link prediction, I suggest to only display the link prediction results as the author did.


