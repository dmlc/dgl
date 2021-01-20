PointNet and PointNet++ for Point Cloud Classification and Segmentation
====

This is a reproduction of the papers
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593).
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413).

# Performance

## Classification
| Model           | Dataset    | Metric   | Score |
|-----------------|------------|----------|-------|
| PointNet        | ModelNet40 | Accuracy | 89.3  |
| PointNet++(SSG) | ModelNet40 | Accuracy | 93.26 |
| PointNet++(MSG) | ModelNet40 | Accuracy | 93.26 |

## Part Segmentation

| Model           | Dataset    | Metric   | Score |
|-----------------|------------|----------|-------|
| PointNet        | ShapeNet   | mIoU     | 83.6  |
| PointNet++(SSG) | ShapeNet   | mIoU     | 84.5  |
| PointNet++(MSG) | ShapeNet   | mIoU     | 84.6  |

# How to Run

For point cloud classification, run with

```python
python train_cls.py
```

For point cloud part-segmentation, run with

```python
python train_partseg.py
```

## To Visualize Part Segmentation in Tensorboard
First ``pip install tensorboard``
then run
```python 
python train_partseg_with_visualization.py
```
To use Tensorboard, run 
``tensorboard --logdir=runs``
