Point Transformer
====

> This model is implemented on August 27, 2021 when there is no official code released.    
Thus we implemented this model based on the code from <https://github.com/qq456cvb/Point-Transformers>.

This is a reproduction of the paper: [Point Transformer](http://arxiv.org/abs/2012.09164).

# Performance
| Task           | Dataset    | Metric   | Score - Paper  | Score - DGL (Adam) | Score - DGL (SGD) | Time(s) - DGL |
|-----------------|------------|----------|------------------|-------------|-------------|-------------------|
| Classification        | ModelNet40 | Accuracy | 93.7   | 92.0        |  91.5        | 117.0          |
| Part Segmentation        | ShapeNet   | mIoU     | 86.6            | 84.3        |  85.1        | 260.0         |

+ Time(s) are the average training time per epoch, measured on EC2 p3.8xlarge instance w/ Tesla V100 GPU.

# How to Run

For point cloud classification, run with

```python
python train_cls.py --opt [sgd/adam]
```

For point cloud part-segmentation, run with

```python
python train_partseg.py --opt [sgd/adam]
```
