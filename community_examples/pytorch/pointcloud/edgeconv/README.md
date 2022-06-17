Dynamic EdgeConv
====

This is a reproduction of the paper [Dynamic Graph CNN for Learning on Point
Clouds](https://arxiv.org/pdf/1801.07829.pdf).

The reproduced experiment is the 40-class classification on the ModelNet40
dataset.  The sampled point clouds are identical to that of
[PointNet](https://github.com/charlesq34/pointnet).

To train and test the model, simply run

```python
python main.py
```

The model currently takes 3 minutes to train an epoch on Tesla V100, and an
additional 17 seconds to run a validation and 20 seconds to run a test.

The best validation performance is 93.5% with a test performance of 91.8%.

## Dependencies

* `h5py`
* `tqdm`
