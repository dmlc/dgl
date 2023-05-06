Gated Graph Convolutional Network(Gated-GCN)
============================================

Residual Gated Graph ConvNets [https://arxiv.org/pdf/1711.07553v2.pdf](https://arxiv.org/pdf/1711.07553v2.pdf)

Contributor: [paoxiaode](https://github.com/paoxiaode)

## Dateset

Task: Graph Property Prediction

|   Dataset   | #Graphs | #Node Feats | #Edge Feats | Metric |
| :---------: | :-----: | :---------: | :---------: | :-----: |
| ogbg-molhiv | 41,127 |      9      |      3      | ROC-AUC |

How to run
----------

```bash
python main.py --epochs 50 --batch-size 32 --num-layers 8
```

## Summary

* ogbg-molhiv: ~0.781
