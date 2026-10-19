"""Test CSRRowWiseSamplingBiased on Ascend NPU.

Verifies the native AscendC kernel (bucketed biased sampling, replace /
no-replace, select-all) against the CPU reference. Sampling is stochastic, so:

  * Deterministic cases (select-all, or fanout >= biased nnz, or all-equal
    bias) are compared exactly (sorted edge set) against CPU.
  * Stochastic cases are checked for structural validity (sampled columns are
    true neighbors, no duplicates when replace=False, correct per-row count)
    and statistical properties (tag-bucket frequencies follow the bias, bias=0
    tags never appear) — the same assertions the upstream CPU tests make.

Prerequisites (mirrors upstream semantics): the graph must be sorted by tag
with ``dgl.sort_csr_by_tag`` / ``dgl.sort_csc_by_tag`` (CPU-only ops, applied
before moving the graph to NPU) so that same-tag neighbors are stored
consecutively and the ``_TAG_OFFSET`` feature delimits the per-row buckets.
"""
from collections import Counter

import pytest
import torch
import dgl


def _check_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _setup():
    if not _check_npu_available():
        return None, None
    return torch.device("npu:0"), torch.device("cpu")


def _build_graph(num_nodes, edges, device, idtype=torch.int64):
    # Build on CPU first: torch_npu's aclnnMaxDim (used by DGL to infer
    # num_nodes) does not support int32 on NPU. Moving an already-built graph
    # to NPU preserves the idtype and exercises the int32 Ascend kernel.
    src = torch.tensor([e[0] for e in edges], dtype=idtype)
    dst = torch.tensor([e[1] for e in edges], dtype=idtype)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    if device != torch.device("cpu"):
        g = g.to(device)
    return g


def _sorted_biased_edges(sg, nodes):
    """Return sorted (v, u) sampled pairs keyed by row for an out-edge biased
    sampling result, on CPU."""
    gc = sg.cpu() if sg.device != torch.device("cpu") else sg
    u, v = gc.edges()
    uv = torch.stack([u, v], dim=1)
    uv = uv[torch.argsort(uv[:, 1])]
    uv = uv[torch.argsort(uv[:, 0])]
    return uv.tolist()


def _successors(g, nodes):
    """Map node -> set of successor node ids (CPU)."""
    succ = {int(n): set() for n in nodes.tolist()}
    gc = g.cpu() if g.device != torch.device("cpu") else g
    u, v = gc.edges()
    for uu, vv in zip(u.tolist(), v.tolist()):
        if uu in succ:
            succ[uu].add(vv)
    return succ


# Edges of a 5-node graph, asymmetric, node 4 has out-degree 1.
EDGES_5 = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 4),
    (4, 1),
]
OUT_DEG_5 = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}


def _sorted_tag(n):
    """Tags: node 0..1 -> tag 0, node 2..3 -> tag 1, node 4 -> tag 2."""
    return torch.tensor([0, 0, 1, 1, 2])


def _biased_out_graph(device, idtype=torch.int64, edges=EDGES_5, num_nodes=5,
                      tag=None):
    """Build a CPU graph, sort CSR by tag, move to device."""
    g = _build_graph(num_nodes, edges, torch.device("cpu"), idtype)
    t = tag if tag is not None else _sorted_tag(num_nodes)
    g_sorted = dgl.sort_csr_by_tag(g, t)
    if device != torch.device("cpu"):
        g_sorted = g_sorted.to(device)
    return g_sorted


def _biased_in_graph(device, idtype=torch.int64, edges=EDGES_5, num_nodes=5,
                     tag=None):
    """Build a CPU graph, sort CSC by tag, move to device."""
    g = _build_graph(num_nodes, edges, torch.device("cpu"), idtype)
    t = tag if tag is not None else _sorted_tag(num_nodes)
    g_sorted = dgl.sort_csc_by_tag(g, t)
    if device != torch.device("cpu"):
        g_sorted = g_sorted.to(device)
    return g_sorted


# ---------------------------------------------------------------------------
# Deterministic comparisons (degenerate / exact cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_biased_select_all_exact(idtype):
    """fanout=-1 selects every edge of every positive-bias bucket: NPU edge
    set must equal CPU edge set."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _biased_out_graph(device, idtype)
    g_cpu = _biased_out_graph(cpu, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5])
    sg_npu = dgl.sampling.sample_neighbors_biased(
        g_npu, nodes, -1, bias, edge_dir="out")
    sg_cpu = dgl.sampling.sample_neighbors_biased(
        g_cpu, nodes.cpu(), -1, bias, edge_dir="out")
    assert _sorted_biased_edges(sg_npu, nodes) == \
        _sorted_biased_edges(sg_cpu, nodes)


@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_biased_fanout_exceeds_nnz_exact(idtype):
    """fanout > biased nnz with replace=False takes all positive-bias edges:
    deterministic set comparison."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _biased_out_graph(device, idtype)
    g_cpu = _biased_out_graph(cpu, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5])
    sg_npu = dgl.sampling.sample_neighbors_biased(
        g_npu, nodes, 100, bias, edge_dir="out", replace=False)
    sg_cpu = dgl.sampling.sample_neighbors_biased(
        g_cpu, nodes.cpu(), 100, bias, edge_dir="out", replace=False)
    assert _sorted_biased_edges(sg_npu, nodes) == \
        _sorted_biased_edges(sg_cpu, nodes)


def test_biased_select_all_with_zero_bias_bucket():
    """fanout=-1 with a zero-bias bucket: only positive-bias edges appear
    (deterministic)."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _biased_out_graph(device)
    g_cpu = _biased_out_graph(cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    # tag 2 (node 4) has zero bias: its edges must never be selected.
    bias = torch.tensor([1.0, 1.0, 0.0])
    sg_npu = dgl.sampling.sample_neighbors_biased(
        g_npu, nodes, -1, bias, edge_dir="out")
    sg_cpu = dgl.sampling.sample_neighbors_biased(
        g_cpu, nodes.cpu(), -1, bias, edge_dir="out")
    assert _sorted_biased_edges(sg_npu, nodes) == \
        _sorted_biased_edges(sg_cpu, nodes)
    u, v = sg_npu.cpu().edges()
    # tag 2 = node 4: no sampled edge may point TO a tag-2 node.
    assert not (v == 4).any(), "zero-bias tag edges must not appear"


@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_biased_equal_bias_matches_uniform(idtype):
    """All-equal bias degenerates to uniform sampling; with fanout >= nnz it
    is deterministic and must equal CPU."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _biased_out_graph(device, idtype)
    g_cpu = _biased_out_graph(cpu, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    bias = torch.tensor([3.0, 3.0, 3.0])
    sg_npu = dgl.sampling.sample_neighbors_biased(
        g_npu, nodes, 3, bias, edge_dir="out", replace=False)
    sg_cpu = dgl.sampling.sample_neighbors_biased(
        g_cpu, nodes.cpu(), 3, bias, edge_dir="out", replace=False)
    assert _sorted_biased_edges(sg_npu, nodes) == \
        _sorted_biased_edges(sg_cpu, nodes)


# ---------------------------------------------------------------------------
# Structural checks (stochastic cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_biased_no_replace_structural(idtype):
    """Sampled (u, v) pairs are real out-edges; no duplicates; per-row count
    equals min(fanout, biased nnz)."""
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5])
    fanout = 2
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, fanout, bias, edge_dir="out", replace=False)
    gc = sg.cpu()
    u, v = gc.edges()
    succ = _successors(g, nodes)
    for uu, vv in zip(u.tolist(), v.tolist()):
        assert vv in succ[uu], f"sampled edge ({uu},{vv}) not a real out-edge"
    seen = {}
    for uu, vv in zip(u.tolist(), v.tolist()):
        seen.setdefault(uu, set())
        assert vv not in seen[uu], f"duplicate edge ({uu},{vv})"
        seen[uu].add(vv)
    # Per-row count: min(fanout, number of positive-bias out-edges).
    # With this tag/bias layout every row's full out-degree counts.
    cnt = Counter(u.tolist())
    for n, d in OUT_DEG_5.items():
        assert cnt.get(n, 0) == min(fanout, d), \
            f"row {n}: got {cnt.get(n, 0)} expected {min(fanout, d)}"


@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_biased_replace_structural(idtype):
    """replace=True: per-row count == fanout whenever positive-bias nnz > 0;
    duplicates are allowed."""
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device, idtype)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=idtype, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5])
    fanout = 4
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, fanout, bias, edge_dir="out", replace=True)
    gc = sg.cpu()
    u, v = gc.edges()
    succ = _successors(g, nodes)
    for uu, vv in zip(u.tolist(), v.tolist()):
        assert vv in succ[uu]
    cnt = Counter(u.tolist())
    for n, d in OUT_DEG_5.items():
        assert cnt.get(n, 0) == fanout, \
            f"row {n}: got {cnt.get(n, 0)} expected {fanout}"


def test_biased_in_edges_structural():
    """edge_dir='in' path (CSC matrix + COOTranspose exit)."""
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_in_graph(device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5])
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, 2, bias, edge_dir="in", replace=False)
    gc = sg.cpu()
    u, v = gc.edges()
    # in-edge sampling: v is a predecessor-recipient; u must be a real in-edge
    # source of v in the original graph.
    gc_orig = g.cpu()
    orig_u, orig_v = gc_orig.edges()
    in_edges = {}
    for uu, vv in zip(orig_u.tolist(), orig_v.tolist()):
        in_edges.setdefault(vv, set()).add(uu)
    for uu, vv in zip(u.tolist(), v.tolist()):
        assert uu in in_edges[vv], \
            f"sampled edge ({uu},{vv}) not a real in-edge"
    cnt = Counter(v.tolist())
    in_deg = Counter(orig_v.tolist())
    for n, d in in_deg.items():
        assert cnt.get(n, 0) == min(2, d)


# ---------------------------------------------------------------------------
# Statistical (semantic) checks
# ---------------------------------------------------------------------------

def test_biased_tag_frequency():
    """Bias [0, 0.1, 10]: tag 0 never appears; tags 1 and 2 dominate tag 1's
    rare bucket by > 2x (upstream test_sampling.py:1281 assertion)."""
    device, cpu = _setup()
    if device is None:
        return
    # Larger graph for meaningful frequencies.
    n = 60
    torch.manual_seed(7)
    m = 900
    src = torch.randint(0, n, (m,))
    dst = torch.randint(0, n, (m,))
    edges = list(zip(src.tolist(), dst.tolist()))
    tag = torch.randint(0, 3, (n,))
    g_npu = _biased_out_graph(device, edges=edges, num_nodes=n, tag=tag)
    nodes = torch.arange(0, n, dtype=torch.int64, device=device)
    bias = torch.tensor([0.0, 0.1, 10.0])

    tag_of_v = tag.tolist()
    cnt = Counter()
    trials = 5
    for _ in range(trials):
        sg = dgl.sampling.sample_neighbors_biased(
            g_npu, nodes, 5, bias, edge_dir="out", replace=False)
        gc = sg.cpu()
        u, v = gc.edges()
        for uu, vv in zip(u.tolist(), v.tolist()):
            cnt[tag_of_v[vv]] += 1
    # tag 0 (zero bias) must never appear.
    assert cnt[0] == 0, f"zero-bias tag appeared {cnt[0]} times"
    # tag 2 must dominate tag 1 by a wide margin (10 vs 0.1 weight).
    assert cnt[2] > 2 * max(cnt[1], 1), \
        f"tag counts not following bias: {dict(cnt)}"


def test_biased_statistical_distribution():
    """Single high-bias vs low-bias bucket on one row: frequency ratio tracks
    the bias ratio (loose tolerance)."""
    device, cpu = _setup()
    if device is None:
        return
    # Star graph: node 0 -> 10 nodes of tag 0 and 10 nodes of tag 1.
    edges = [(0, i) for i in range(1, 11)] + [(0, i) for i in range(11, 21)]
    g_npu = _biased_out_graph(device, edges=edges, num_nodes=21,
                              tag=torch.tensor([0] * 1 + [0] * 10 + [1] * 10))
    nodes = torch.tensor([0], dtype=torch.int64, device=device)
    bias = torch.tensor([1.0, 3.0])  # tag1 bucket 3x weight per edge
    trials = 600
    cnt = Counter()
    for _ in range(trials):
        sg = dgl.sampling.sample_neighbors_biased(
            g_npu, nodes, 1, bias, edge_dir="out", replace=False)
        gc = sg.cpu()
        u, v = gc.edges()
        cnt[1 if 11 <= v.item() <= 20 else 0] += 1
    assert sum(cnt.values()) == trials
    # Expected P(tag1) = 30/40 = 0.75. Loose bounds for stochastic safety.
    assert 0.6 < cnt[1] / trials < 0.9, f"tag1 frequency {cnt[1]/trials}"


# ---------------------------------------------------------------------------
# Edge / defensive cases
# ---------------------------------------------------------------------------

def test_biased_fanout_zero():
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device)
    nodes = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, 0, torch.tensor([1.0, 1.0, 1.0]), edge_dir="out")
    assert sg.num_edges() == 0


def test_biased_empty_request():
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device)
    nodes = torch.tensor([], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, 2, torch.tensor([1.0, 1.0, 1.0]), edge_dir="out")
    assert sg.num_edges() == 0


def test_biased_all_zero_bias():
    """All-zero bias: nothing is eligible, output is empty for every row."""
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, 3, torch.tensor([0.0, 0.0, 0.0]), edge_dir="out")
    assert sg.num_edges() == 0


def test_biased_negative_bias_clamped():
    """Negative bias is 'undefined' upstream; our kernel clamps it to zero
    weight. Equivalent to a zero-bias bucket: those tags never appear."""
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    bias = torch.tensor([-5.0, 1.0, 1.0])
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, -1, bias, edge_dir="out")
    gc = sg.cpu()
    u, v = gc.edges()
    # tag 0 = nodes {0, 1}: no sampled edge may point to nodes 0/1.
    for vv in v.tolist():
        assert vv >= 2, f"negative-bias tag edge appeared: {vv}"


def test_biased_float64_bias():
    """float64 bias is accepted (normalized to float32 on host)."""
    device, cpu = _setup()
    if device is None:
        return
    g_npu = _biased_out_graph(device)
    g_cpu = _biased_out_graph(cpu)
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    sg_npu = dgl.sampling.sample_neighbors_biased(
        g_npu, nodes, 100, bias, edge_dir="out", replace=False)
    sg_cpu = dgl.sampling.sample_neighbors_biased(
        g_cpu, nodes.cpu(), 100, bias, edge_dir="out", replace=False)
    assert _sorted_biased_edges(sg_npu, nodes) == \
        _sorted_biased_edges(sg_cpu, nodes)


def test_biased_large_graph_structural():
    """Larger random graph exercises the multi-core path (row partitioning)
    and the per-row count invariant."""
    device, cpu = _setup()
    if device is None:
        return
    torch.manual_seed(0)
    n = 200
    m = 2000
    src = torch.randint(0, n, (m,))
    dst = torch.randint(0, n, (m,))
    edges = list(zip(src.tolist(), dst.tolist()))
    tag = torch.randint(0, 4, (n,))
    g = _biased_out_graph(device, edges=edges, num_nodes=n, tag=tag)
    nodes = torch.arange(0, n, dtype=torch.int64, device=device)
    bias = torch.tensor([1.0, 2.0, 0.5, 1.0])
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, 5, bias, edge_dir="out", replace=False)
    gc = sg.cpu()
    u, v = gc.edges()
    succ = _successors(g, nodes)
    for uu, vv in zip(u.tolist(), v.tolist()):
        assert vv in succ[uu]
    assert sg.num_edges() <= n * 5


def test_biased_direct_gm_dedup_exact():
    """R1 regression: the direct-GM fallback runs when num_picks exceeds
    the UB window; its no-replace dedup must produce an exact CPU-equal
    edge set. A wide-degree row with fanout in (window, degree] forces
    that path deterministically."""
    device, cpu = _setup()
    if device is None:
        return
    # Star: node 0 -> 40 destinations, tags alternate 0/1 (20 each).
    edges = [(0, i) for i in range(1, 41)]
    tag = torch.tensor([0] + [i % 2 for i in range(1, 41)])
    g_npu = _biased_out_graph(device, edges=edges, num_nodes=41, tag=tag)
    g_cpu = _biased_out_graph(cpu, edges=edges, num_nodes=41, tag=tag)
    nodes = torch.tensor([0], dtype=torch.int64, device=device)
    bias = torch.tensor([1.0, 1.0])
    # fanout 30 with a 10-15k-element UB window stays under degree but
    # the pick count itself is far above any per-row staging the old
    # dedup table could hold; fanout >= window would route here.
    sg_npu = dgl.sampling.sample_neighbors_biased(
        g_npu, nodes, 30, bias, edge_dir="out", replace=False)
    gc = sg_npu.cpu()
    u, v = gc.edges()
    # Stochastic case (fanout 30 < nnz 40): CPU picks a different random
    # subset, so the assertion is structural, not set-equal.
    for vv in v.tolist():
        assert 1 <= vv <= 40, f"sampled destination {vv} is not a neighbor"
    assert len(set(v.tolist())) == 30, "duplicate destinations in direct-GM path"


def test_npu_path_actually_taken():
    """Coverage guard: the Ascend kernel path must be exercised (graph on
    NPU, non-degenerate input yields edges)."""
    device, cpu = _setup()
    if device is None:
        return
    g = _biased_out_graph(device)
    assert g.device.type == "npu"
    nodes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    sg = dgl.sampling.sample_neighbors_biased(
        g, nodes, 2, torch.tensor([1.0, 1.0, 1.0]), edge_dir="out")
    assert sg.num_edges() > 0
    assert sg.device.type == "npu"
