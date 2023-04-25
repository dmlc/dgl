import backend as F

import dgl
import dgl.nn
import numpy as np
import pytest
import torch as th
from dgl import DGLError
from dgl.base import DGLWarning
from dgl.geometry import farthest_point_sampler, neighbor_matching
from utils import parametrize_idtype
from utils.graph_cases import get_cases


def test_fps():
    N = 1000
    batch_size = 5
    sample_points = 10
    x = th.tensor(np.random.uniform(size=(batch_size, int(N / batch_size), 3)))
    ctx = F.ctx()
    if F.gpu_ctx():
        x = x.to(ctx)
    res = farthest_point_sampler(x, sample_points)
    assert res.shape[0] == batch_size
    assert res.shape[1] == sample_points
    assert res.sum() > 0


def test_fps_start_idx():
    N = 1000
    batch_size = 5
    sample_points = 10
    x = th.tensor(np.random.uniform(size=(batch_size, int(N / batch_size), 3)))
    ctx = F.ctx()
    if F.gpu_ctx():
        x = x.to(ctx)
    res = farthest_point_sampler(x, sample_points, start_idx=0)
    assert th.any(res[:, 0] == 0)


def _test_knn_common(device, algorithm, dist, exclude_self):
    x = th.randn(8, 3).to(device)
    kg = dgl.nn.KNNGraph(3)
    if dist == "euclidean":
        d = th.cdist(x, x).to(F.cpu())
    else:
        x = x + th.randn(1).item()
        tmp_x = x / (1e-5 + F.sqrt(F.sum(x * x, dim=1, keepdims=True)))
        d = 1 - F.matmul(tmp_x, tmp_x.T).to(F.cpu())

    def check_knn(g, x, start, end, k, exclude_self, check_indices=True):
        assert g.device == x.device
        g = g.to(F.cpu())
        for v in range(start, end):
            src, _ = g.in_edges(v)
            src = set(src.numpy())
            assert len(src) == k
            if check_indices:
                i = v - start
                src_ans = set(
                    th.topk(
                        d[start:end, start:end][i],
                        k + (1 if exclude_self else 0),
                        largest=False,
                    )[1].numpy()
                    + start
                )
                if exclude_self:
                    # remove self
                    src_ans.remove(v)
                assert src == src_ans

    def check_batch(g, k, expected_batch_info):
        assert F.array_equal(g.batch_num_nodes(), F.tensor(expected_batch_info))
        assert F.array_equal(
            g.batch_num_edges(), k * F.tensor(expected_batch_info)
        )

    # check knn with 2d input
    g = kg(x, algorithm, dist, exclude_self)
    check_knn(g, x, 0, 8, 3, exclude_self)
    check_batch(g, 3, [8])

    # check knn with 3d input
    g = kg(x.view(2, 4, 3), algorithm, dist, exclude_self)
    check_knn(g, x, 0, 4, 3, exclude_self)
    check_knn(g, x, 4, 8, 3, exclude_self)
    check_batch(g, 3, [4, 4])

    # check segmented knn
    # there are only 2 edges per node possible when exclude_self with 3 nodes in the segment
    # and this test case isn't supposed to warn, so limit it when exclude_self is True
    adjusted_k = 3 - (1 if exclude_self else 0)
    kg = dgl.nn.SegmentedKNNGraph(adjusted_k)
    g = kg(x, [3, 5], algorithm, dist, exclude_self)
    check_knn(g, x, 0, 3, adjusted_k, exclude_self)
    check_knn(g, x, 3, 8, adjusted_k, exclude_self)
    check_batch(g, adjusted_k, [3, 5])

    # check k > num_points
    kg = dgl.nn.KNNGraph(10)
    with pytest.warns(DGLWarning):
        g = kg(x, algorithm, dist, exclude_self)
    # there are only 7 edges per node possible when exclude_self with 8 nodes total
    adjusted_k = 8 - (1 if exclude_self else 0)
    check_knn(g, x, 0, 8, adjusted_k, exclude_self)
    check_batch(g, adjusted_k, [8])

    with pytest.warns(DGLWarning):
        g = kg(x.view(2, 4, 3), algorithm, dist, exclude_self)
    # there are only 3 edges per node possible when exclude_self with 4 nodes per segment
    adjusted_k = 4 - (1 if exclude_self else 0)
    check_knn(g, x, 0, 4, adjusted_k, exclude_self)
    check_knn(g, x, 4, 8, adjusted_k, exclude_self)
    check_batch(g, adjusted_k, [4, 4])

    kg = dgl.nn.SegmentedKNNGraph(5)
    with pytest.warns(DGLWarning):
        g = kg(x, [3, 5], algorithm, dist, exclude_self)
    # there are only 2 edges per node possible when exclude_self in the segment with
    # only 3 nodes, and the current implementation reduces k for all segments
    # in that case
    adjusted_k = 3 - (1 if exclude_self else 0)
    check_knn(g, x, 0, 3, adjusted_k, exclude_self)
    check_knn(g, x, 3, 8, adjusted_k, exclude_self)
    check_batch(g, adjusted_k, [3, 5])

    # check k == 0
    # that's valid for exclude_self, but -1 is not, so check -1 instead for exclude_self
    adjusted_k = 0 - (1 if exclude_self else 0)
    kg = dgl.nn.KNNGraph(adjusted_k)
    with pytest.raises(DGLError):
        g = kg(x, algorithm, dist, exclude_self)
    kg = dgl.nn.SegmentedKNNGraph(adjusted_k)
    with pytest.raises(DGLError):
        g = kg(x, [3, 5], algorithm, dist, exclude_self)

    # check empty
    x_empty = th.tensor([])
    kg = dgl.nn.KNNGraph(3)
    with pytest.raises(DGLError):
        g = kg(x_empty, algorithm, dist, exclude_self)
    kg = dgl.nn.SegmentedKNNGraph(3)
    with pytest.raises(DGLError):
        g = kg(x_empty, [3, 5], algorithm, dist, exclude_self)

    # check all coincident points
    x = th.zeros((20, 3)).to(device)
    kg = dgl.nn.KNNGraph(3)
    g = kg(x, algorithm, dist, exclude_self)
    # different algorithms may break the tie differently, so don't check the indices
    check_knn(g, x, 0, 20, 3, exclude_self, False)
    check_batch(g, 3, [20])

    # check all coincident points
    kg = dgl.nn.SegmentedKNNGraph(3)
    g = kg(x, [4, 7, 5, 4], algorithm, dist, exclude_self)
    # different algorithms may break the tie differently, so don't check the indices
    check_knn(g, x, 0, 4, 3, exclude_self, False)
    check_knn(g, x, 4, 11, 3, exclude_self, False)
    check_knn(g, x, 11, 16, 3, exclude_self, False)
    check_knn(g, x, 16, 20, 3, exclude_self, False)
    check_batch(g, 3, [4, 7, 5, 4])


@pytest.mark.parametrize(
    "algorithm", ["bruteforce-blas", "bruteforce", "kd-tree"]
)
@pytest.mark.parametrize("dist", ["euclidean", "cosine"])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_knn_cpu(algorithm, dist, exclude_self):
    _test_knn_common(F.cpu(), algorithm, dist, exclude_self)


@pytest.mark.parametrize(
    "algorithm", ["bruteforce-blas", "bruteforce", "bruteforce-sharemem"]
)
@pytest.mark.parametrize("dist", ["euclidean", "cosine"])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_knn_cuda(algorithm, dist, exclude_self):
    if not th.cuda.is_available():
        return
    _test_knn_common(F.cuda(), algorithm, dist, exclude_self)


@pytest.mark.parametrize("num_points", [8, 64, 256, 1024])
def test_knn_sharedmem_large(num_points):
    if not th.cuda.is_available():
        return
    x = th.randn(num_points, 5, device="cuda")
    y = th.randn(num_points, 5, device="cuda")
    k = 4

    def ground_truth(x, y, k):
        dist = (
            th.sum(x * x, dim=1)
            + th.sum(y * y, dim=1).unsqueeze(-1)
            - 2 * th.mm(y, x.T)
        )
        ret = th.topk(dist, k, dim=-1, largest=False)[1]
        return th.sort(ret, dim=-1)[0]

    gt = ground_truth(x, y, k)
    actual = th.sort(
        dgl.functional.knn(
            k, x, [num_points], y, [num_points], algorithm="bruteforce-sharemem"
        )[1].reshape(-1, k),
        -1,
    )[0]
    assert th.all(actual == gt).item()


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["dglgraph"]))
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("relabel", [True, False])
def test_edge_coarsening(idtype, g, weight, relabel):
    num_nodes = g.num_nodes()
    g = dgl.to_bidirected(g)
    g = g.astype(idtype).to(F.ctx())
    edge_weight = None
    if weight:
        edge_weight = F.abs(F.randn((g.num_edges(),))).to(F.ctx())
    node_labels = neighbor_matching(g, edge_weight, relabel_idx=relabel)
    unique_ids, counts = th.unique(node_labels, return_counts=True)
    num_result_ids = unique_ids.size(0)

    # shape correct
    assert node_labels.shape == (g.num_nodes(),)

    # all nodes marked
    assert F.reduce_sum(node_labels < 0).item() == 0

    # number of unique node ids correct.
    assert num_result_ids >= num_nodes // 2 and num_result_ids <= num_nodes

    # each unique id has <= 2 nodes
    assert F.reduce_sum(counts > 2).item() == 0

    # if two nodes have the same id, they must be neighbors
    idxs = F.arange(0, num_nodes, idtype)
    for l in unique_ids:
        l = l.item()
        idx = idxs[(node_labels == l)]
        if idx.size(0) == 2:
            u, v = idx[0].item(), idx[1].item()
            assert g.has_edges_between(u, v)


if __name__ == "__main__":
    test_fps()
    test_fps_start_idx()
    test_knn()
    test_knn_sharedmem_large()
