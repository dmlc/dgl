PCT
====

This is a reproduction of the paper: [PCT: Point cloud transformer](http://arxiv.org/abs/2012.09688).

# Performance
| Task           | Dataset    | Metric   | Score - Paper  | Score - DGL (Adam) | Time(s) - DGL |
|-----------------|------------|----------|------------------|-------------|-------------------|
| Classification        | ModelNet40 | Accuracy | 93.2   | 92.1      | 740.0          |
| Part Segmentation        | ShapeNet   | mIoU     | 86.4            | 85.6       | 390.0         |

+ Time(s) are the average training time per epoch, measured on EC2 g4dn.12xlarge instance w/ Tesla T4 GPU.
+ We run the code with the preprocessing used in [PointNet++](../pointnet). We can only get 84.5 for classification if we use the preprocessing described in the paper:
    > During training, a random translation in [−0.2, 0.2], a random anisotropic scaling in [0.67, 1.5] and a random input dropout were applied to augment the input data.


# How to Run

For point cloud classification, run with

```python
python train_cls.py
```

For point cloud part-segmentation, run with

```python
python train_partseg.py
```
