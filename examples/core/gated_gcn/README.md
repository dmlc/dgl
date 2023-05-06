Gated Graph ConvNet (GatedGCN)
============================================


* paper link: <https://arxiv.org/abs/2003.00982.pdf>


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
