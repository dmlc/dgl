Gated Graph ConvNet (GatedGCN)
==============================

* paper link: [https://arxiv.org/abs/2003.00982.pdf](https://arxiv.org/abs/2003.00982.pdf)

## Dataset

Task: Graph Property Prediction

|   Dataset   | #Graphs | #Node Feats | #Edge Feats | Metric |
| :---------: | :-----: | :---------: | :---------: | :-----: |
| ogbg-molhiv | 41,127 |      9      |      3      | ROC-AUC |

How to run
----------

```bash
python3 train.py --dataset ogbg-molhiv --num_gpus 0 --num_epochs 50
```

## Summary

* ogbg-molhiv: ~0.781
