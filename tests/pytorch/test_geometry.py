import backend as F
import dgl.nn
import dgl
import numpy as np
import pytest
import torch as th
from dgl.geometry.pytorch import FarthestPointSampler
from dgl.geometry import graph_matching
from test_utils import parametrize_dtype
from test_utils.graph_cases import get_cases


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


@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['dglgraph']))
@pytest.mark.parametrize('weight', [True, False])
def test_graph_matching(idtype, g, weight):
    g = dgl.to_bidirected(g)
    g = g.astype(idtype).to(F.ctx())
    edge_weight = None
    if weight:
        edge_weight = F.abs(F.randn((g.num_edges(),))).to(F.ctx())
    node_labels = graph_matching(g, edge_weight)

    assert node_labels.shape == (g.num_nodes(),)      # shape correct
    assert F.reduce_sum(node_labels < 0).item() == 0  # all nodes marked


if __name__ == '__main__':
    test_fps()
    test_knn()
