"""Test CSRRowWiseSampling on Ascend NPU.

Verifies the native AscendC kernels (uniform + weighted, replace / no-replace,
select-all) against a CPU reference. Sampling is stochastic, so:

  * Deterministic cases (select-all, or fanout >= degree) are compared exactly
    (sorted edge set) against CPU.
  * Stochastic cases are checked for structural validity (sampled columns are
    true neighbors, no duplicates when replace=False, correct per-row count)
    and statistical properties (uniform / weighted distribution over many
    seeds).

DGL's ``sample_neighbors`` samples in-edges (predecessors) by default, so both
the NPU and CPU reference operate on the same CSC view.
"""
import pytest
import torch
import dgl


def _check_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _setup():
    if not _check_npu_available():
        return None, None
    return torch.device("npu:0"), torch.device("cpu")


# A handful of small graphs used across tests.
def _build_graph(num_nodes, edges, device, idtype=torch.int64):
    # Build on CPU first: torch_npu's aclnnMaxDim (used by DGL to infer
    # num_nodes) does not support int32 on NPU. Moving an already-built graph
    # to NPU preserves the idtype and exercises the int32 Ascend kernel.
    src = torch.tensor([e[0] for e in edges], dtype=idtype)
    dst = torch.tensor([e[1] for e in edges], dtype=idtype)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    if device != torch.device("cpu"):
        g = g.to(device)
    return g.formats("csc")


# edges for a 5-node graph (asymmetric, with a node of degree 0 and 1).
EDGES_5 = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 4),
    (4, 1),
]
# in-degree (predecessor count) per node for EDGES_5: node 4 has 1, none has 0.
IN_DEG_5 = {0: 2, 1: 2, 2: 2, 3: 3, 4: 1}


def _uv(g):
    """Return (u, v) CPU tensors of a graph's edges.

    The graph may live on NPU. Move to CPU first (COOSort_, needed for
    ``order='srcdst'``, is only implemented on CPU; and CSR-format graphs do
    not support the default ``eid`` order). The sampling under test has
    already happened on NPU by the time this is called.
    """
    gc = g.cpu() if g.device != torch.device("cpu") else g
    return gc.edges(order="srcdst")


def _sorted_edges(g):
    """Return sorted (u, v) pairs of a graph's edges, on CPU."""
    u, v = _uv(g)
    uv = torch.stack([u, v], dim=1)
    uv = uv[torch.argsort(uv[:, 1])]
    uv = uv[torch.argsort(uv[:, 0])]
    return uv.tolist()


def _predecessors(g, nodes):
    """Map node -> set of predecessor node ids (CPU)."""
    pred = {int(n): set() for n in nodes.tolist()}
    u, v = _uv(g)
    for uu, vv in zip(u.tolist(), v.tolist()):
        if vv in pred:
            pred[vv].add(uu)
    return pred


def _set_prob(g, prob_tensor, name="p"):
    """Attach a probability tensor as an edge feature and return its name.

    ``dgl.sampling.sample_neighbors`` expects ``prob`` to be a feature name
    (str), not a bare tensor; passing a tensor is silently treated as a failed
    feature lookup and degenerates to uniform sampling.
    """
    g.edata[name] = prob_tensor
    return name


# ---------------------------------------------------------------------------
# Uniform sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_uniform_select_all_exact(idtype):
    """fanout=-1 is deterministic: NPU edge set must equal CPU edge set."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device, idtype)
    g_cpu = _build_graph(5, EDGES_5, cpu, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    nodes_cpu = nodes.cpu()

    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1)
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes_cpu, -1)
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_uniform_no_replace_structural(idtype):
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    fanout = 2
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=False)
    u, v = _uv(sg)
    pred = _predecessors(g, nodes)
    # Each sampled (u, v): u is a predecessor of v.
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert uu in pred[vv], f"sampled edge ({uu},{vv}) not a real in-edge"
    # No duplicate (u, v) per row v.
    seen = {}
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        seen.setdefault(vv, set())
        assert uu not in seen[vv], f"duplicate edge ({uu},{vv})"
        seen[vv].add(uu)
    # Per-row count == min(fanout, in-degree).
    from collections import Counter
    cnt = Counter(v.cpu().tolist())
    for n, d in IN_DEG_5.items():
        assert cnt.get(n, 0) == min(fanout, d), \
            f"node {n}: got {cnt.get(n, 0)} expected {min(fanout, d)}"


def test_uniform_replace_structural():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    fanout = 4
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=True)
    u, v = _uv(sg)
    pred = _predecessors(g, nodes)
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert uu in pred[vv]
    # replace=True: count == fanout for every node with degree > 0.
    from collections import Counter
    cnt = Counter(v.cpu().tolist())
    for n, d in IN_DEG_5.items():
        if d > 0:
            assert cnt.get(n, 0) == fanout, \
                f"node {n}: got {cnt.get(n, 0)} expected {fanout}"


def test_uniform_fanout_exceeds_degree_exact():
    """fanout > degree, no-replace: must pick ALL in-edges (deterministic)."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device)
    g_cpu = _build_graph(5, EDGES_5, cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, 100, replace=False)
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes.cpu(), 100, replace=False)
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


def test_uniform_empty_request():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 2)
    assert sg.num_edges() == 0


def test_uniform_fanout_zero():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 0)
    assert sg.num_edges() == 0


def test_uniform_statistical():
    """Over many seeds, each predecessor is picked with ~equal frequency."""
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([3], dtype=torch.int64, device=device)
    # Node 3 has 3 predecessors: edges (0,3),(1,3),(2,3) -> preds {0,1,2}.
    fanout = 1
    trials = 600
    from collections import Counter
    counts = Counter()
    for _ in range(trials):
        sg = dgl.sampling.sample_neighbors(g, nodes, fanout, replace=False)
        u, v = _uv(sg)
        counts[u.item()] += 1
    # Expect ~200 each (3 preds, 600 trials). Allow generous tolerance.
    assert sum(counts.values()) == trials, counts
    for pred in (0, 1, 2):
        assert 120 < counts[pred] < 280, f"pred {pred}: {counts}"


# ---------------------------------------------------------------------------
# Weighted sampling (float32 probability)
# ---------------------------------------------------------------------------

def test_weighted_select_all_exact():
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device)
    g_cpu = _build_graph(5, EDGES_5, cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    # float32 prob; zero out a few edges so select-all filters them.
    ne = g_npu.num_edges()
    prob = torch.full((ne,), 0.5, dtype=torch.float32)
    prob[0] = 0.0
    prob[3] = 0.0
    pname_n = _set_prob(g_npu, prob.to(device))
    pname_c = _set_prob(g_cpu, prob)
    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1, prob=pname_n)
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes.cpu(), -1, prob=pname_c)
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


def test_weighted_no_replace_structural():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    ne = g.num_edges()
    prob = (torch.rand(ne, dtype=torch.float32) + 0.1).to(device)
    pname = _set_prob(g, prob)
    fanout = 2
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, prob=pname, replace=False)
    u, v = _uv(sg)
    pred = _predecessors(g, nodes)
    for uu, vv in zip(u.tolist(), v.tolist()):
        assert uu in pred[vv]
    seen = {}
    for uu, vv in zip(u.tolist(), v.tolist()):
        seen.setdefault(vv, set())
        assert uu not in seen[vv], f"duplicate ({uu},{vv})"
        seen[vv].add(uu)


def test_weighted_replace_structural():
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    ne = g.num_edges()
    prob = (torch.rand(ne, dtype=torch.float32) + 0.1).to(device)
    pname = _set_prob(g, prob)
    fanout = 3
    sg = dgl.sampling.sample_neighbors(g, nodes, fanout, prob=pname, replace=True)
    u, v = _uv(sg)
    pred = _predecessors(g, nodes)
    for uu, vv in zip(u.tolist(), v.tolist()):
        assert uu in pred[vv]
    from collections import Counter
    cnt = Counter(v.tolist())
    for n, d in IN_DEG_5.items():
        if d > 0:
            assert cnt.get(n, 0) == fanout


def test_weighted_all_zero_prob_row():
    """All-zero probability yields 0 picks (no crash)."""
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    ne = g.num_edges()
    prob_zero = torch.zeros(ne, dtype=torch.float32).to(device)
    pname = _set_prob(g, prob_zero)
    sg = dgl.sampling.sample_neighbors(g, nodes, 2, prob=pname, replace=False)
    assert sg.num_edges() == 0


def test_weighted_statistical():
    """Higher-prob predecessor is picked more often (replace)."""
    device, cpu = _setup()
    if device is None:
        return
    g = _build_graph(5, EDGES_5, device)
    nodes = torch.tensor([3], dtype=torch.int64, device=device)
    # Node 3 predecessors: edges (0,3),(1,3),(2,3) -> preds {0,1,2}.
    ne = g.num_edges()
    prob = torch.full((ne,), 0.01, dtype=torch.float32)
    # Edge IDs follow EDGES_5 construction order; make edge (0,3) dominant.
    target_eid = EDGES_5.index((0, 3))
    prob[target_eid] = 0.99
    prob = prob.to(device)
    pname = _set_prob(g, prob)
    fanout = 1
    trials = 400
    hit0 = 0
    for _ in range(trials):
        sg = dgl.sampling.sample_neighbors(g, nodes, fanout, prob=pname, replace=True)
        u, v = _uv(sg)
        if int(u.item()) == 0:
            hit0 += 1
    # prob 0.99 vs 0.01 -> expect ~99% of picks to be predecessor 0.
    assert hit0 > trials * 0.9, hit0


def test_weighted_float64_prob_cast():
    """float64 probability is cast to float32 internally and still works
    (select-all must match the CPU float64 reference)."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _build_graph(5, EDGES_5, device)
    g_cpu = _build_graph(5, EDGES_5, cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    ne = g_npu.num_edges()
    prob = torch.full((ne,), 0.5, dtype=torch.float64)
    prob[0] = 0.0
    prob[3] = 0.0
    pname_n = _set_prob(g_npu, prob.to(device))
    pname_c = _set_prob(g_cpu, prob)
    sg_npu = dgl.sampling.sample_neighbors(g_npu, nodes, -1, prob=pname_n)
    sg_cpu = dgl.sampling.sample_neighbors(g_cpu, nodes.cpu(), -1, prob=pname_c)
    assert _sorted_edges(sg_npu) == _sorted_edges(sg_cpu)


# ---------------------------------------------------------------------------
# Larger / stress
# ---------------------------------------------------------------------------

def test_large_graph_uniform():
    device, cpu = _setup()
    if device is None:
        return
    torch.manual_seed(0)
    n = 200
    m = 2000
    src = torch.randint(0, n, (m,))
    dst = torch.randint(0, n, (m,))
    g = dgl.graph((src.to(device), dst.to(device)), num_nodes=n).formats("csc")
    nodes = torch.arange(0, n, dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors(g, nodes, 5, replace=False)
    u, v = _uv(sg)
    pred = _predecessors(g, nodes)
    for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()):
        assert uu in pred[vv]
    assert sg.num_edges() <= n * 5
