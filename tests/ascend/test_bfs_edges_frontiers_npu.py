"""
Test the BFSEdgesFrontiers Ascend operator through the DGL high-level API
(``dgl.bfs_edges_generator``) on NPU.
"""

import sys

import dgl

import numpy as np
import pytest
import torch
from dgl import heterograph_index

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _check_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _setup():
    """Return (npu_device, cpu_device) or (None, None) if NPU is unavailable."""
    if not _check_npu_available():
        return None, None
    return torch.device("npu:0"), torch.device("cpu")


def _make_graph_from_csr(indptr, indices, num_nodes, dtype, device, eid=None):
    """Build a DGLGraph whose unit-graph adjacency is exactly the given CSR.

    ``eid=None`` -> empty edge-id array (hasEid=0 kernel path, edge ids equal
    to the CSR position); ``eid=list`` -> explicit edge-id mapping (hasEid=1).
    """
    if eid is None:
        eids = torch.tensor([], dtype=dtype)
    else:
        eids = torch.tensor(eid, dtype=dtype)
    gidx = heterograph_index.create_unitgraph_from_csr(
        1,
        num_nodes,
        num_nodes,
        torch.tensor(indptr, dtype=dtype),
        torch.tensor(indices, dtype=dtype),
        eids,
        ["csr"],
    )
    g = dgl.DGLGraph(gidx)
    g = g.formats("csr")
    return g.to(device)


def _run_bfs_edges(g, source):
    return list(dgl.bfs_edges_generator(g, source))


def _assert_frontiers_equal(name, npu_out, cpu_out):
    assert len(npu_out) == len(
        cpu_out
    ), f"{name}: frontier count mismatch NPU={len(npu_out)} CPU={len(cpu_out)}"
    for i, (a, b) in enumerate(zip(npu_out, cpu_out)):
        a_cpu = a.cpu() if hasattr(a, "cpu") else a
        b_cpu = b.cpu() if hasattr(b, "cpu") else b
        assert torch.equal(a_cpu, b_cpu), (
            f"{name}: frontier[{i}] bit-exact mismatch\n"
            f"  NPU={a_cpu.tolist()}\n  CPU={b_cpu.tolist()}"
        )


def _assert_raises_dgl_error(fn, cls=None):
    """Assert ``fn`` raises dgl.DGLError; optionally check errflag class."""
    with pytest.raises(dgl.DGLError) as exc_info:
        fn()
    if cls is not None:
        msg = str(exc_info.value)
        assert (
            f"class={cls}" in msg
        ), f"expected errflag class={cls} but got: {msg[:300]}"


# ---------------------------------------------------------------------------
# tiling-key static prediction (transcribed from calc_bfs_tiling_params)
# ---------------------------------------------------------------------------

_UB_KEY0_BUDGET = 96 * 1024
_KEY1_MIN_ROWS = 8192


def _predict_tiling_key(num_rows, nnz, has_eid, elem_size):
    ub_bytes = (
        (num_rows + 1) * elem_size
        + nnz * elem_size * (1 + has_eid)
        + num_rows * 4
    )
    if ub_bytes <= _UB_KEY0_BUDGET:
        return 0
    if num_rows >= _KEY1_MIN_ROWS and nnz < 2 * num_rows:
        return 1
    return 2


# ---------------------------------------------------------------------------
# graph specs (migrated from cases_def.py GRAPHS; invalid CSR for L2)
# ---------------------------------------------------------------------------

GRAPHS = {
    # L0-08/L0-09/L1-10 tree 0->1,0->2,1->3,1->4,2->5,2->6 (num_rows=7)
    "tree7": ([0, 2, 4, 6, 6, 6, 6, 6], [1, 2, 3, 4, 5, 6], 7),
    # L0-10/L1-22 duplicate-edge graph 0->1,0->1,1->2 (num_rows=3)
    "multi3": ([0, 2, 3, 3], [1, 1, 2], 3),
    # L0-11 self-loop graph 0->0,0->1 (num_rows=2)
    "self2": ([0, 2, 2], [0, 1], 2),
    # L0-12 all-source-covered graph 0->1,0->2 (num_rows=3)
    "n12": ([0, 2, 2, 2], [1, 2], 3),
    # L1-12 multi-component 0->1,2->3,4->5 (num_rows=6)
    "comp6": ([0, 1, 1, 2, 2, 3, 3], [1, 3, 5], 6),
    # L1-14/L1-13 chain 0->1->2->3
    "chain4": ([0, 1, 2, 3, 3], [1, 2, 3], 4),
    # L1-16 nnz=0 all-isolated, num_rows=3
    "iso3": ([0, 0, 0, 0], [], 3),
    # L1-17 empty graph num_rows=0
    "empty_graph": ([0], [], 0),
    # L1-18 num_rows=1 minimal graph
    "numrows1": ([0, 0], [], 1),
    # L2-05 indices out-of-range graph
    "l2_04": ([0, 1, 2, 3, 3], [1, 2, 9], 4),
    # L2-07 indptr non-monotonic
    "l2_06": ([0, 2, 1, 3, 3], [1, 2, 3], 4),
}


def _build_chain(num_rows):
    """Chain 0->1->...->num_rows-1; nnz=num_rows-1."""
    indptr = np.concatenate(
        [np.arange(num_rows, dtype=np.int64), [num_rows - 1]]
    )
    indices = np.arange(1, num_rows, dtype=np.int64)
    return indptr, indices


def _build_star(num_rows):
    """Star 0->1..num_rows-1; node0 holds all nnz=num_rows-1 edges."""
    indptr = np.full(num_rows + 1, num_rows - 1, dtype=np.int64)
    indptr[0] = 0
    indices = np.arange(1, num_rows, dtype=np.int64)
    return indptr, indices


def _resolve_l1_graph(name):
    """Resolve a graph name used by the L1 table: small graphs come from the
    GRAPHS dict, generated large graphs (chain1024 / star1024) are built on
    the fly.  Returns (indptr, indices, num_nodes)."""
    if name in GRAPHS:
        return GRAPHS[name]
    if name == "chain1024":
        indptr, indices = _build_chain(1024)
        return indptr, indices, int(indptr.size - 1)
    if name == "star1024":
        indptr, indices = _build_star(1024)
        return indptr, indices, int(indptr.size - 1)
    raise KeyError("unknown L1 graph: %s" % name)


def _build_rand1k():
    """1K random graph (fixed seed, same as cases_def.py RAND1K_SPEC)."""
    spec = {
        "num_rows": 1024,
        "avg_deg": 4,
        "seed": 20260919,
        "orphan_node": 1023,
    }
    rng = np.random.default_rng(spec["seed"])
    num_rows = spec["num_rows"]
    orphan = spec["orphan_node"]
    degs = rng.poisson(spec["avg_deg"], size=num_rows)
    degs[orphan] = 0
    rows = []
    for u in range(num_rows):
        d = int(degs[u])
        if d <= 0:
            rows.append(np.array([], dtype=np.int64))
            continue
        tgt = rng.integers(0, num_rows, size=d)
        tgt = rng.permutation(tgt)
        rows.append(tgt.astype(np.int64))
    nnz = int(sum(r.size for r in rows))
    if nnz:
        indices = np.concatenate(rows).astype(np.int64)
    else:
        indices = np.array([], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(
        np.int64
    )
    src = rng.choice(num_rows, size=8, replace=False).tolist()
    if orphan not in src:
        src[-1] = orphan
    return indptr, indices, src


def _rand1k_eid_map(nnz):
    """L1-04 random non-contiguous eid map (fixed seed, range [0, 4*num_rows))."""
    rng = np.random.default_rng(20260919 + 1)
    return rng.integers(0, 4 * 1024, size=nnz).astype(np.int64)


def _build_rand_wide(spec):
    """Key2 main graph: num_rows nodes, avg_deg avg out-degree, fixed seed."""
    rng = np.random.default_rng(spec["seed"])
    num_rows = spec["num_rows"]
    degs = rng.poisson(spec["avg_deg"], size=num_rows).clip(0, 50)
    rows = []
    for u in range(num_rows):
        d = int(degs[u])
        if d <= 0:
            rows.append(np.array([], dtype=np.int64))
        else:
            rows.append(rng.integers(0, num_rows, size=d).astype(np.int64))
    indices = (
        np.concatenate(rows).astype(np.int64)
        if sum(r.size for r in rows)
        else np.array([], dtype=np.int64)
    )
    indptr = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(
        np.int64
    )
    source = np.array(spec["sources"], dtype=np.int64)
    return indptr, indices, source


def _build_circulant(spec):
    """Key2 error-path base graph (circulant, strongly connected)."""
    num_rows = spec["num_rows"]
    steps = spec["steps"]
    edges = []
    for u in range(num_rows):
        for s in steps:
            edges.append((u + s) % num_rows)
    indices = np.array(edges, dtype=np.int64)
    indptr = np.arange(
        0, (num_rows + 1) * len(steps), len(steps), dtype=np.int64
    )
    source = np.array(spec["source"], dtype=np.int64)
    return indptr, indices, source


def _k2rand_eid_map(nnz):
    """WB-K2-04 random non-contiguous eid map (fixed seed, range [0, 4*num_rows))."""
    rng = np.random.default_rng(20260920 + 1)
    return rng.integers(0, 4 * 100000, size=nnz).astype(np.int64)


def _inject_error(indptr, indices, source, err_kind):
    """Inject an invalid CSR/source field; semantics follow kernel SetErr."""
    indptr = indptr.copy()
    indices = indices.copy()
    source = source.copy()
    num_rows = int(indptr.size - 1)
    if err_kind == "src_oob":
        source = np.array([num_rows], dtype=np.int64)
    elif err_kind == "src_neg":
        source = np.array([-3], dtype=np.int64)
    elif err_kind == "idx_oob":
        indices[0] = num_rows
    elif err_kind == "indptr_nonmono":
        k = num_rows // 2
        indptr[k], indptr[k + 1] = indptr[k + 1], indptr[k]
    elif err_kind == "nnz_mismatch":
        indptr[-1] = int(indptr[-1]) - 1
    else:
        raise KeyError("unknown err_kind: %s" % err_kind)
    return indptr, indices, source


# ---------------------------------------------------------------------------
# L0 functional anchors (edges; both int32 and int64)
# ---------------------------------------------------------------------------

L0_CASES = [
    # (case_id, graph, source, eid, description)
    (
        "L0-08",
        "tree7",
        [0],
        None,
        "edges tree no eid (first layer not counted)",
    ),
    (
        "L0-09",
        "tree7",
        [0],
        [5, 3, 1, 0, 4, 2],
        "edges tree shuffled explicit eid",
    ),
    ("L0-10", "multi3", [0], None, "edges duplicate-edge first tree edge"),
    ("L0-11", "self2", [0], None, "edges self-loop non-tree edge"),
    ("L0-12", "n12", [0, 1, 2], None, "edges all-source-covered empty output"),
]


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "case_id,graph,source,eid,desc",
    [pytest.param(c, g, s, e, d, id=c) for c, g, s, e, d in L0_CASES],
)
def test_bfs_edges_l0_npu(case_id, graph, source, eid, desc, dtype):
    # CPU reference first (independent of NPU availability).
    indptr, indices, num_nodes = GRAPHS[graph]
    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(
        indptr, indices, num_nodes, dtype, cpu, eid=eid
    )
    cpu_ref = _run_bfs_edges(g_cpu, source)

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(
        indptr, indices, num_nodes, dtype, device, eid=eid
    )
    npu_out = _run_bfs_edges(g_npu, source)
    torch.npu.synchronize()
    _assert_frontiers_equal(f"{case_id}/{desc}", npu_out, cpu_ref)


# ---------------------------------------------------------------------------
# L1 functional extension (edges; both int32 and int64)
# ---------------------------------------------------------------------------

L1_CASES = [
    # (case_id, graph, source, eid, description)
    ("L1-06", "chain1024", [0], None, "deep chain 1024 edges no eid"),
    ("L1-08", "star1024", [0], None, "star single-layer no eid"),
    ("L1-10", "tree7", [0], [2, 2, 2, 2, 2, 2], "dup eid passthrough"),
    ("L1-12", "comp6", [0, 2, 4], None, "multi-component"),
    ("L1-14", "chain4", [], None, "empty source (S=0 early exit)"),
    ("L1-16", "iso3", [1], None, "nnz=0 empty output"),
    ("L1-17", "empty_graph", [], None, "empty graph"),
    ("L1-18", "numrows1", [0], None, "num_rows=1"),
    ("L1-22", "multi3", [0, 2], [9, 7, 9], "eid + dup edge + orphan source"),
]


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "case_id,graph,source,eid,desc",
    [pytest.param(c, g, s, e, d, id=c) for c, g, s, e, d in L1_CASES],
)
def test_bfs_edges_l1_npu(case_id, graph, source, eid, desc, dtype):
    indptr, indices, num_nodes = _resolve_l1_graph(graph)
    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(
        indptr, indices, num_nodes, dtype, cpu, eid=eid
    )
    cpu_ref = _run_bfs_edges(g_cpu, source)

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(
        indptr, indices, num_nodes, dtype, device, eid=eid
    )
    npu_out = _run_bfs_edges(g_npu, source)
    torch.npu.synchronize()
    _assert_frontiers_equal(f"{case_id}/{desc}", npu_out, cpu_ref)


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_bfs_edges_l1_rand1k_npu(dtype):
    """L1-03/L1-04: 1K random graph, 8 deterministic sources.  eid variants:
    no eid (L1-03) and random non-contiguous eid map (L1-04), both dtypes."""
    indptr, indices, src = _build_rand1k()
    num_nodes = int(indptr.size - 1)
    source = src
    cpu = torch.device("cpu")
    eid_map = _rand1k_eid_map(int(indices.size)).tolist()

    for variant, eid in (("noeid", None), ("randmap", eid_map)):
        g_cpu = _make_graph_from_csr(
            indptr, indices, num_nodes, dtype, cpu, eid=eid
        )
        cpu_ref = _run_bfs_edges(g_cpu, source)

        device, _ = _setup()
        if device is None:
            pytest.skip("NPU not available")
        g_npu = _make_graph_from_csr(
            indptr, indices, num_nodes, dtype, device, eid=eid
        )
        npu_out = _run_bfs_edges(g_npu, source)
        torch.npu.synchronize()
        _assert_frontiers_equal(f"L1-03/04/rand1k/{variant}", npu_out, cpu_ref)


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_bfs_edges_l1_20_eid_variant_diff_npu(dtype):
    """L1-20: identity eid vs empty eid variant diff on tree7 (S4).  Both
    variants must produce equal output (eid == CSR position in both cases),
    matching the CPU reference."""
    indptr, indices, num_nodes = GRAPHS["tree7"]
    cpu = torch.device("cpu")
    identity = list(range(len(indices)))

    cpu_ident = _run_bfs_edges(
        _make_graph_from_csr(
            indptr, indices, num_nodes, dtype, cpu, eid=identity
        ),
        [0],
    )
    cpu_empty = _run_bfs_edges(
        _make_graph_from_csr(indptr, indices, num_nodes, dtype, cpu, eid=None),
        [0],
    )
    _assert_frontiers_equal("L1-20/cpu-identity-vs-empty", cpu_ident, cpu_empty)

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    npu_ident = _run_bfs_edges(
        _make_graph_from_csr(
            indptr, indices, num_nodes, dtype, device, eid=identity
        ),
        [0],
    )
    npu_empty = _run_bfs_edges(
        _make_graph_from_csr(
            indptr, indices, num_nodes, dtype, device, eid=None
        ),
        [0],
    )
    torch.npu.synchronize()
    _assert_frontiers_equal("L1-20/npu-identity-vs-empty", npu_ident, npu_empty)
    _assert_frontiers_equal("L1-20/npu-identity-vs-cpu", npu_ident, cpu_ident)


# ---------------------------------------------------------------------------
# L2 exception cases (edges)
# ---------------------------------------------------------------------------

L2_CALL_RAISE = [
    # (case_id, graph, source, errflag_class or None, description)
    ("L2-03", "chain4", [4], 3, "source >= num_rows"),
    ("L2-05", "l2_04", [0], 4, "indices out of range"),
    ("L2-07", "l2_06", [0], 1, "indptr non-monotonic"),
    ("L2-12", "empty_graph", [0], None, "empty graph + non-empty source"),
    ("L2-14", "chain4", [-1], 3, "negative source"),
]


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "case_id,graph,source,cls,desc",
    [pytest.param(c, g, s, cl, d, id=c) for c, g, s, cl, d in L2_CALL_RAISE],
)
def test_bfs_edges_l2_call_raises_npu(case_id, graph, source, cls, desc, dtype):
    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    indptr, indices, num_nodes = GRAPHS[graph]
    g_npu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, device)
    _assert_raises_dgl_error(lambda: _run_bfs_edges(g_npu, source), cls=cls)
    torch.npu.synchronize()  # no crash residue after the error


def test_bfs_edges_l2_10_eid_length_raises_npu():
    """L2-10: eid length mismatch (6 indices vs 5 eids) is rejected at DGL
    graph-construction time (CSR CheckValidity data-length check)."""
    with pytest.raises(dgl.DGLError):
        gidx = heterograph_index.create_unitgraph_from_csr(
            1,
            7,
            7,
            torch.tensor([0, 2, 4, 6, 6, 6, 6, 6], dtype=torch.int64),
            torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64),
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            ["csr"],
        )
        dgl.DGLGraph(gidx)


# ---------------------------------------------------------------------------
# White-box: tiling-key coverage (Key1/Key2) + errflag class code
# ---------------------------------------------------------------------------

WB_TILING_CASES = [
    # (case_id, graph_builder, source, expected_key, dtype, eid, desc)
    (
        "WB-K1-04",
        "chain20000",
        [0],
        1,
        torch.int32,
        None,
        "Key1 deep chain edges int32",
    ),
    (
        "WB-K1-05",
        "chain20000",
        [0],
        1,
        torch.int64,
        None,
        "Key1 deep chain edges int64",
    ),
    (
        "WB-K2-03",
        "k2rand",
        "spec",
        2,
        torch.int32,
        None,
        "Key2 main graph edges no eid",
    ),
    (
        "WB-K2-04",
        "k2rand",
        "spec",
        2,
        torch.int64,
        "RANDOM_MAP",
        "Key2 main graph edges random eid",
    ),
]


def _build_wb_graph(name):
    """Return (indptr, indices, source_or_None) for a white-box graph spec."""
    if name == "chain20000":
        indptr, indices = _build_chain(20000)
        return indptr, indices, None
    if name == "k2rand":
        spec = {
            "num_rows": 100000,
            "avg_deg": 4.5,
            "seed": 20260920,
            "sources": [0, 33333, 66666, 99999],
        }
        return _build_rand_wide(spec)
    raise KeyError("unknown white-box graph: %s" % name)


@pytest.mark.parametrize(
    "case_id,name,source,expected_key,dtype,eid,desc",
    [
        pytest.param(c, n, s, k, dt, e, d, id=c)
        for c, n, s, k, dt, e, d in WB_TILING_CASES
    ],
)
def test_bfs_edges_whitebox_tiling_key_npu(
    case_id, name, source, expected_key, dtype, eid, desc
):
    """Construct a graph whose scale statically falls into the intended tiling
    key (1/2) and verify bit-exact NPU output against the CPU reference."""
    indptr, indices, _ = _build_wb_graph(name)
    num_nodes = int(indptr.size - 1)
    nnz = int(indices.size)
    has_eid = 1 if (eid is not None) else 0
    elem_size = 4 if dtype == torch.int32 else 8
    actual_key = _predict_tiling_key(num_nodes, nnz, has_eid, elem_size)
    assert actual_key == expected_key, (
        f"{case_id}: constructed graph has tiling key {actual_key}, "
        f"expected {expected_key} (num_rows={num_nodes} nnz={nnz} has_eid={has_eid})"
    )
    src = source if source != "spec" else [0]
    eid_arr = None if eid is None else _k2rand_eid_map(nnz).tolist()

    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(
        indptr, indices, num_nodes, dtype, cpu, eid=eid_arr
    )
    cpu_ref = _run_bfs_edges(g_cpu, src)

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(
        indptr, indices, num_nodes, dtype, device, eid=eid_arr
    )
    npu_out = _run_bfs_edges(g_npu, src)
    torch.npu.synchronize()
    _assert_frontiers_equal(f"{case_id}/{desc}", npu_out, cpu_ref)


# errflag class code on a Key2-size (multi-block) base graph (edges entry)
_WB_ERR_BASE = {
    "num_rows": 10000,
    "steps": (1, 2, 3),
    "seed": None,
    "source": [0],
}


def test_bfs_edges_whitebox_errflag_class_npu():
    """WB-K2-ERR-06: device-side validation class 4 (indices out of range) on
    a Key2-size (multi-block) graph through the edges entry."""
    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    indptr, indices, source = _build_circulant(_WB_ERR_BASE)
    indptr, indices, source = _inject_error(indptr, indices, source, "idx_oob")
    num_nodes = int(indptr.size - 1)
    g_npu = _make_graph_from_csr(
        indptr, indices, num_nodes, torch.int32, device
    )
    _assert_raises_dgl_error(
        lambda: _run_bfs_edges(g_npu, source.tolist()), cls=4
    )
    torch.npu.synchronize()  # no crash residue after the error


# ---------------------------------------------------------------------------
# __main__ runner (mirrors the style of the other tests/ascend/test_*_npu.py)
# ---------------------------------------------------------------------------


def _collect_tests():
    tests = []

    for dtype in (torch.int32, torch.int64):
        dt = "int32" if dtype is torch.int32 else "int64"
        for c, g, s, e, d in L0_CASES:
            tests.append(
                (
                    f"L0/{c}/{dt}",
                    lambda c=c, g=g, s=s, e=e, d=d, dt=dtype: test_bfs_edges_l0_npu(
                        c, g, s, e, d, dt
                    ),
                )
            )
        for c, g, s, e, d in L1_CASES:
            tests.append(
                (
                    f"L1/{c}/{dt}",
                    lambda c=c, g=g, s=s, e=e, d=d, dt=dtype: test_bfs_edges_l1_npu(
                        c, g, s, e, d, dt
                    ),
                )
            )
        tests.append(
            (
                f"L1/rand1k/{dt}",
                lambda dt=dtype: test_bfs_edges_l1_rand1k_npu(dt),
            )
        )
        tests.append(
            (
                f"L1/L1-20/{dt}",
                lambda dt=dtype: test_bfs_edges_l1_20_eid_variant_diff_npu(dt),
            )
        )
        for c, g, s, cl, d in L2_CALL_RAISE:
            tests.append(
                (
                    f"L2/{c}/{dt}",
                    lambda c=c, g=g, s=s, cl=cl, d=d, dt=dtype: test_bfs_edges_l2_call_raises_npu(
                        c, g, s, cl, d, dt
                    ),
                )
            )
    tests.append(
        ("L2/L2-10/construct", test_bfs_edges_l2_10_eid_length_raises_npu)
    )
    for c, n, s, k, dt, e, d in WB_TILING_CASES:
        tests.append(
            (
                f"WB/{c}",
                lambda c=c, n=n, s=s, k=k, dt=dt, e=e, d=d: test_bfs_edges_whitebox_tiling_key_npu(
                    c, n, s, k, dt, e, d
                ),
            )
        )
    tests.append(("WB/WB-K2-ERR-06", test_bfs_edges_whitebox_errflag_class_npu))
    return tests


if __name__ == "__main__":
    if not _check_npu_available():
        print("NPU not available; aborting.", flush=True)
        sys.exit(0)

    failures = 0
    skipped = 0
    errors = 0
    cases = _collect_tests()
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS [{name}]")
        except AssertionError as e:
            print(f"  FAIL [{name}] {e}")
            failures += 1
        except BaseException as e:  # noqa: BLE001
            # pytest.skip() raises Skipped (BaseException subclass)
            ename = type(e).__name__.lower()
            if "skipped" in ename:
                print(f"  SKIP [{name}] {e}")
                skipped += 1
            else:
                print(f"  ERROR[{name}] {type(e).__name__}: {e}")
                errors += 1
    total = len(cases)
    passed = total - failures - skipped - errors
    print(
        f"\nResults: {passed}/{total} passed, {failures} failed, "
        f"{skipped} skipped, {errors} errored",
        flush=True,
    )
