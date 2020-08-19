import torch as th
import dgl.nn
from dgl.geometry.pytorch import FarthestPointSampler
import backend as F
import numpy as np

def test_fps():
    N = 1000
    batch_size = 5
    sample_points = 10
    x = th.tensor(np.random.uniform(size=(batch_size, int(N/batch_size), 3)))
    ctx = F.ctx()
    if F.gpu_ctx():
        x = x.to(ctx)
    fps = FarthestPointSampler(sample_points)
    res = fps(x)
    assert res.shape[0] == batch_size
    assert res.shape[1] == sample_points
    assert res.sum() > 0

def test_knn():
    x = th.randn(8, 3)
    kg = dgl.nn.KNNGraph(3)
    d = th.cdist(x, x)

    def check_knn(g, x, start, end):
        for v in range(start, end):
            src, _ = g.in_edges(v)
            src = set(src.numpy())
            i = v - start
            src_ans = set(th.topk(d[start:end, start:end][i], 3, largest=False)[1].numpy() + start)
            assert src == src_ans

    g = kg(x)
    check_knn(g, x, 0, 8)

    g = kg(x.view(2, 4, 3))
    check_knn(g, x, 0, 4)
    check_knn(g, x, 4, 8)

    kg = dgl.nn.SegmentedKNNGraph(3)
    g = kg(x, [3, 5])
    check_knn(g, x, 0, 3)
    check_knn(g, x, 3, 8)

if __name__ == '__main__':
    test_fps()
    test_knn()
