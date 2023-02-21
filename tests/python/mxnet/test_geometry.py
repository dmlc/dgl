import backend as F
import mxnet as mx
import numpy as np

from dgl.geometry import farthest_point_sampler


def test_fps():
    N = 1000
    batch_size = 5
    sample_points = 10
    x = mx.nd.array(
        np.random.uniform(size=(batch_size, int(N / batch_size), 3))
    )
    ctx = F.ctx()
    if F.gpu_ctx():
        x = x.as_in_context(ctx)
    res = farthest_point_sampler(x, sample_points)
    assert res.shape[0] == batch_size
    assert res.shape[1] == sample_points
    assert res.sum() > 0


if __name__ == "__main__":
    test_fps()
