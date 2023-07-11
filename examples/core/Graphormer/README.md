Graphormer
==============================

## Introduction

* Graphormer is a Transformer model designed for graph-structured data, which encodes the structural information of a graph into the standard Transformer. Specifically, Graphormer utilizes Degree Encoding to measure the importance of nodes, Spatial Encoding and Path Encoding to measure the relation between node pairs. The former plus the node features serve as input to Graphormer, while the latter acts as bias terms in the self-attention module.

* paper link: [https://arxiv.org/abs/2106.05234](https://arxiv.org/abs/2106.05234)

## Requirements
- accelerate
- transformers
- ogb

## Dataset

Task: Graph Property Prediction

|   Dataset   | #Graphs | #Node Feats | #Edge Feats | Metric  |
| :---------: | :-----: | :---------: | :---------: | :-----: |
| ogbg-molhiv | 41,127  |      9      |      3      | ROC-AUC |

How to run
----------

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 train.py
```
> **_NOTE:_**  The script will automatically download weights pre-trained on PCQM4Mv2. To reproduce the same result, set the total batch size to 64.

## Summary

* ogbg-molhiv (pretrained on PCQM4Mv2): ~0.791
