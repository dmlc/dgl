Point Transformer
====

This is a reproduction of the paper: [Point Transformer](http://arxiv.org/abs/2012.09164).

# Performance
| Task           | Dataset    | Metric   | Score - Paper  | Score - DGL | Time(s) - DGL |
|-----------------|------------|----------|------------------|-------------|-------------------|
| Classification        | ModelNet40 | Accuracy | 93.7   | 90.4        | 101.0          |
<!-- | Part Segmentation        | ShapeNet   | mIoU     | 86.6            | 83.6        | 234.0         | -->

+ Time(s) are the average training time per epoch, measured on EC2 g4dn.4xlarge instance w/ Tesla T4 GPU.

# How to Run

For point cloud classification, run with

```python
python train_cls.py
```

<!-- For point cloud part-segmentation, run with

```python
python train_partseg.py
``` -->
