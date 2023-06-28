Graphormer
==============================

* paper link: [https://arxiv.org/abs/2106.05234](https://arxiv.org/abs/2106.05234)

## Requirements
- accelerate
- transformers
- ogb

## Dataset

Task: Graph Property Prediction

|   Dataset   | #Graphs | #Node Feats | #Edge Feats | Metric |
| :---------: | :-----: | :---------: | :---------: | :-----: |
| ogbg-molhiv | 41,127 |      9      |      3      | ROC-AUC |

How to run
----------

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 train.py
```

## Summary

* ogbg-molhiv (pretrained on PCQM4Mv2): ~0.791
