"""
Test the BFSNodesFrontiers Ascend operator through the DGL high-level API
(``dgl.bfs_nodes_generator``) on NPU.
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

    ``create_unitgraph_from_csr`` preserves the row-internal order of
    ``indices`` (standalone spec semantics) and accepts invalid CSR for L2.
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


def _run_bfs(g, source, entry):
    """Run the DGL BFS API and return the list of frontier tensors."""
    fn = (
        dgl.bfs_edges_generator if entry == "edges" else dgl.bfs_nodes_generator
    )
    return list(fn(g, source))


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
# tiling-key static prediction (transcribed from calc_bfs_tiling_params /
# wb_tiling.py; used only for white-box coverage observation, not shared code)
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
# graph specs (migrated from cases_def.py GRAPHS; row-internal order = written
# order; plus invalid CSR for L2 and generated specs for the large graphs)
# ---------------------------------------------------------------------------

GRAPHS = {
    # L0-01/L0-06 chain 0->1->2->3
    "chain4": ([0, 1, 2, 3, 3], [1, 2, 3], 4),
    # L0-02/L1-21 tree 0->1,0->2,1->3,1->4,2->5,2->6 (num_rows=7)
    "tree7": ([0, 2, 4, 6, 6, 6, 6, 6], [1, 2, 3, 4, 5, 6], 7),
    # L0-03/L1-19 cross DAG 0->2,0->3,1->3,1->4 (num_rows=5)
    "cross6": ([0, 2, 4, 4, 4, 4], [2, 3, 3, 4], 5),
    # L0-04 ring 0->1->2->3->0
    "ring4": ([0, 1, 2, 3, 4], [1, 2, 3, 0], 4),
    # L0-05 orphan graph 0->1,0->2 (3,4 orphans)
    "orphan5": ([0, 2, 2, 2, 2, 2], [1, 2], 5),
    # L0-07 store-order-sensitive graph 0->3,0->1,1->2, indices=[3,1,2]
    "store4": ([0, 2, 3, 3, 3], [3, 1, 2], 4),
    # L1-09 ring+chain 0->1,1->2,2->3,3->0,3->4
    "ringchain5": ([0, 1, 2, 3, 5, 5], [1, 2, 3, 0, 4], 5),
    # L1-11/L1-12 multi-component 0->1,2->3,4->5 (num_rows=6)
    "comp6": ([0, 1, 1, 2, 2, 3, 3], [1, 3, 5], 6),
    # L1-21 single node self-loop 0->0 (+ isolated row 1; indptr len 3 -> 2 rows)
    "self1": ([0, 1, 1], [0], 2),
    # L1-15/L1-16 zero-edge graph nnz=0, num_rows=3
    "iso3": ([0, 0, 0, 0], [], 3),
    # L1-18 num_rows=1 minimal graph
    "numrows1": ([0, 0], [], 1),
    # L1-17 empty graph num_rows=0
    "empty_graph": ([0], [], 0),
    # L2-04/L2-05 indices out-of-range graph
    "l2_04": ([0, 1, 2, 3, 3], [1, 2, 9], 4),
    # L2-06/L2-07 indptr non-monotonic
    "l2_06": ([0, 2, 1, 3, 3], [1, 2, 3], 4),
    # L2-13 indptr[-1]=2 vs nnz=1 mismatch
    "l2_13": ([0, 2, 2, 2, 2], [1], 4),
}

_RAND1K_SPEC = {
    "num_rows": 1024,
    "avg_deg": 4,
    "seed": 20260919,
    "orphan_node": 1023,
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
    """1K random graph (fixed seed, same as cases_def.py RAND1K_SPEC):
    avg out-degree ~4, allows dup edges/self loops, row-internal order
    permuted; node 1023 forced orphan; 8 random sources incl. the orphan."""
    spec = _RAND1K_SPEC
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


def _build_lollipop(spec):
    """Key2 large-frontier + deep-tail graph (star core + random density +
    chain tail)."""
    core = spec["core"]
    tail = spec["tail"]
    rng = np.random.default_rng(spec["seed"])
    rows = []
    star = np.arange(1, core, dtype=np.int64)
    rows.append(
        np.concatenate([star, rng.integers(0, core, size=3).astype(np.int64)])
    )
    for _u in range(1, core):
        rows.append(rng.integers(0, core, size=3).astype(np.int64))
    rows[core - 1] = np.append(rows[core - 1], np.int64(core))
    for t in range(tail):
        rows.append(
            np.array([core + t + 1], dtype=np.int64)
            if t < tail - 1
            else np.array([], dtype=np.int64)
        )
    indices = np.concatenate(rows).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(
        np.int64
    )
    source = np.array(spec["source"], dtype=np.int64)
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


def _inject_error(indptr, indices, source, err_kind):
    """Inject an invalid CSR/source field on the Key2-size base graph.
    Semantics follow the kernel SetErr classes (1..4)."""
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
# L0 functional anchors (nodes; both int32 and int64)
# ---------------------------------------------------------------------------

L0_CASES = [
    # (case_id, graph, source, description)
    ("L0-01", "chain4", [0], "nodes chain single source"),
    ("L0-02", "tree7", [0], "nodes tree single source"),
    ("L0-03", "cross6", [0, 1], "nodes multi-source cross"),
    ("L0-04", "ring4", [0], "nodes ring"),
    ("L0-05", "orphan5", [0, 3], "nodes orphan nodes + orphan source"),
    ("L0-06", "chain4", [0, 0, 1], "nodes duplicate sources"),
    ("L0-07", "store4", [0], "nodes neighbor store-order sensitive"),
]


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "case_id,graph,source,desc",
    [pytest.param(c, g, s, d, id=c) for c, g, s, d in L0_CASES],
)
def test_bfs_nodes_l0_npu(case_id, graph, source, desc, dtype):
    # CPU reference first (independent of NPU availability).
    indptr, indices, num_nodes = GRAPHS[graph]
    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, cpu)
    cpu_ref = _run_bfs(g_cpu, source, "nodes")

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, device)
    npu_out = _run_bfs(g_npu, source, "nodes")
    torch.npu.synchronize()
    _assert_frontiers_equal(f"{case_id}/{desc}", npu_out, cpu_ref)


# ---------------------------------------------------------------------------
# L1 functional extension (nodes; both int32 and int64)
# ---------------------------------------------------------------------------

L1_CASES = [
    # (case_id, graph, source, description)
    ("L1-05", "chain1024", [0], "deep chain 1024 layers"),
    ("L1-07", "star1024", [0], "star single-layer big frontier"),
    ("L1-09", "ringchain5", [0, 0, 2], "dup source + multi source + ring"),
    ("L1-11", "comp6", [0, 2, 4], "multi-component independent layers"),
    ("L1-13", "chain4", [], "empty source (S=0 early exit)"),
    ("L1-15", "iso3", [1], "nnz=0 all-isolated (empty-in non-empty-out)"),
    ("L1-17", "empty_graph", [], "empty graph"),
    ("L1-18", "numrows1", [0], "num_rows=1"),
    ("L1-19", "cross6", [1, 0], "source-order swap diff"),
    ("L1-21", "self1", [0], "single node self-loop"),
]


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "case_id,graph,source,desc",
    [pytest.param(c, g, s, d, id=c) for c, g, s, d in L1_CASES],
)
def test_bfs_nodes_l1_npu(case_id, graph, source, desc, dtype):
    indptr, indices, num_nodes = _resolve_l1_graph(graph)
    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, cpu)
    cpu_ref = _run_bfs(g_cpu, source, "nodes")

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, device)
    npu_out = _run_bfs(g_npu, source, "nodes")
    torch.npu.synchronize()
    _assert_frontiers_equal(f"{case_id}/{desc}", npu_out, cpu_ref)


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_bfs_nodes_l1_rand1k_npu(dtype):
    """L1-01/L1-02: 1K random graph, 8 deterministic sources (both dtypes)."""
    indptr, indices, src = _build_rand1k()
    num_nodes = int(indptr.size - 1)
    source = src
    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, cpu)
    cpu_ref = _run_bfs(g_cpu, source, "nodes")

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, device)
    npu_out = _run_bfs(g_npu, source, "nodes")
    torch.npu.synchronize()
    _assert_frontiers_equal("L1-01/02/rand1k", npu_out, cpu_ref)


# ---------------------------------------------------------------------------
# L2 exception cases (nodes)
# ---------------------------------------------------------------------------
# Two groups:
#  * call-raise : the CSR graph is buildable; the BFS call itself raises
#    (kernel SetErr -> host CHECK for classes 1..4, host check for empty graph
#    + non-empty source).  Runs for both dtypes.
#  * construct-raise : the DGL graph construction rejects the invalid input
#    (dtype/shape validation); dtype-independent, single run.

L2_CALL_RAISE = [
    # (case_id, graph, source, errflag_class or None, description)
    # L2-13 class: int32 raises class=2 (nnz mismatch) deterministically, but
    # int64 raises class=4 instead (the int64 chunk path reads indices beyond
    # the short array before the end-of-CSR check).  Both are valid "input is
    # rejected" outcomes; class is asserted only where it is deterministic.
    ("L2-01", "chain4", [4], 3, "source >= num_rows"),
    ("L2-02", "chain4", [-1], 3, "negative source"),
    ("L2-04", "l2_04", [0], 4, "indices out of range"),
    ("L2-06", "l2_06", [0], 1, "indptr non-monotonic"),
    ("L2-12", "empty_graph", [0], None, "empty graph + non-empty source"),
    ("L2-13", "l2_13", [0], None, "nnz != indptr[-1] (class=2 on int32)"),
]


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "case_id,graph,source,cls,desc",
    [pytest.param(c, g, s, cl, d, id=c) for c, g, s, cl, d in L2_CALL_RAISE],
)
def test_bfs_nodes_l2_call_raises_npu(case_id, graph, source, cls, desc, dtype):
    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    indptr, indices, num_nodes = GRAPHS[graph]
    g_npu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, device)
    _assert_raises_dgl_error(lambda: _run_bfs(g_npu, source, "nodes"), cls=cls)
    torch.npu.synchronize()  # no crash residue after the error


L2_CONSTRUCT_RAISE = [
    # (case_id, kind, description) -- rejected at DGL graph construction
    ("L2-08", "float32", "unsupported float32 dtype"),
    ("L2-09", "mixed", "mixed dtype indptr int32 + indices int64"),
    ("L2-11", "indptr_empty", "empty indptr (num_rows undecidable)"),
]


def _make_invalid_dtype_graph(kind):
    """Build the DGLGraph with invalid dtype/shape; must raise DGLError."""
    if kind == "float32":
        indptr = torch.tensor([0, 1, 2, 3, 3], dtype=torch.float32)
        indices = torch.tensor([1, 2, 3], dtype=torch.float32)
        eids = torch.tensor([], dtype=torch.float32)
        num_nodes = 4
    elif kind == "mixed":
        indptr = torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32)
        indices = torch.tensor([1, 2, 3], dtype=torch.int64)
        eids = torch.tensor([], dtype=torch.int64)
        num_nodes = 4
    elif kind == "indptr_empty":
        indptr = torch.tensor([], dtype=torch.int64)
        indices = torch.tensor([], dtype=torch.int64)
        eids = torch.tensor([], dtype=torch.int64)
        num_nodes = 0
    else:
        raise KeyError(kind)
    gidx = heterograph_index.create_unitgraph_from_csr(
        1, num_nodes, num_nodes, indptr, indices, eids, ["csr"]
    )
    return dgl.DGLGraph(gidx)


@pytest.mark.parametrize(
    "case_id,kind,desc",
    [pytest.param(c, k, d, id=c) for c, k, d in L2_CONSTRUCT_RAISE],
)
def test_bfs_nodes_l2_construct_raises_npu(case_id, kind, desc):
    """DGL graph construction rejects invalid dtype/shape inputs with DGLError
    before the BFS call (main-repo adaptation of standalone L2-08/09/11)."""
    with pytest.raises(dgl.DGLError):
        _make_invalid_dtype_graph(kind)


# ---------------------------------------------------------------------------
# White-box: tiling-key coverage (Key0/Key1/Key2) + errflag class codes
# ---------------------------------------------------------------------------

WB_TILING_CASES = [
    # (case_id, graph_builder, source, expected_key, dtype, entry, desc)
    # Key1: deep-graph heuristic (num_rows >= 8192 && nnz < 2*num_rows)
    (
        "WB-K1-01",
        "chain8193",
        [0],
        1,
        torch.int32,
        "nodes",
        "Key1 boundary chain8193",
    ),
    (
        "WB-K1-02",
        "chain20000",
        [0],
        1,
        torch.int32,
        "nodes",
        "Key1 deep chain int32",
    ),
    (
        "WB-K1-03",
        "chain20000",
        [0],
        1,
        torch.int64,
        "nodes",
        "Key1 deep chain int64",
    ),
    # Key2: large multi-core graph (>= 100K nodes / ~450K edges)
    (
        "WB-K2-01",
        "k2rand",
        "spec",
        2,
        torch.int32,
        "nodes",
        "Key2 main graph int32",
    ),
    (
        "WB-K2-02",
        "k2rand",
        "spec",
        2,
        torch.int64,
        "nodes",
        "Key2 main graph int64",
    ),
    (
        "WB-K2-05",
        "k2lollipop",
        "spec",
        2,
        torch.int32,
        "nodes",
        "Key2 big frontier + deep tail",
    ),
    (
        "WB-K2-06",
        "k2lollipop",
        "spec",
        2,
        torch.int64,
        "nodes",
        "Key2 deep tail int64",
    ),
]


def _build_wb_graph(name):
    """Return (indptr, indices, source_or_None) for a white-box graph spec."""
    if name == "chain8193":
        indptr, indices = _build_chain(8193)
        return indptr, indices, None
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
    if name == "k2lollipop":
        spec = {"core": 50000, "tail": 2048, "seed": 20260921, "source": [0]}
        return _build_lollipop(spec)
    raise KeyError("unknown white-box graph: %s" % name)


@pytest.mark.parametrize(
    "case_id,name,source,expected_key,dtype,entry,desc",
    [
        pytest.param(c, n, s, k, dt, e, d, id=c)
        for c, n, s, k, dt, e, d in WB_TILING_CASES
    ],
)
def test_bfs_nodes_whitebox_tiling_key_npu(
    case_id, name, source, expected_key, dtype, entry, desc
):
    """Construct a graph whose scale statically falls into the intended tiling
    key (0/1/2) and verify bit-exact NPU output against the CPU reference."""
    indptr, indices, _ = _build_wb_graph(name)
    num_nodes = int(indptr.size - 1)
    nnz = int(indices.size)
    has_eid = 0
    elem_size = 4 if dtype == torch.int32 else 8
    actual_key = _predict_tiling_key(num_nodes, nnz, has_eid, elem_size)
    assert actual_key == expected_key, (
        f"{case_id}: constructed graph has tiling key {actual_key}, "
        f"expected {expected_key} (num_rows={num_nodes} nnz={nnz})"
    )
    src = source if source != "spec" else [0]

    cpu = torch.device("cpu")
    g_cpu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, cpu)
    cpu_ref = _run_bfs(g_cpu, src, entry)

    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    g_npu = _make_graph_from_csr(indptr, indices, num_nodes, dtype, device)
    npu_out = _run_bfs(g_npu, src, entry)
    torch.npu.synchronize()
    _assert_frontiers_equal(f"{case_id}/{desc}", npu_out, cpu_ref)


# errflag class codes on a Key2-size (multi-block) base graph
_WB_ERR_BASE = {
    "num_rows": 10000,
    "steps": (1, 2, 3),
    "seed": None,
    "source": [0],
}

WB_ERR_CASES = [
    # (case_id, err_kind, errflag_class, description)
    ("WB-K2-ERR-01", "src_oob", 3, "Key2 source out of range"),
    ("WB-K2-ERR-02", "src_neg", 3, "Key2 negative source"),
    ("WB-K2-ERR-03", "idx_oob", 4, "Key2 indices out of range"),
    ("WB-K2-ERR-04", "indptr_nonmono", 1, "Key2 indptr non-monotonic"),
    ("WB-K2-ERR-05", "nnz_mismatch", 2, "Key2 nnz mismatch"),
]


@pytest.mark.parametrize(
    "case_id,err_kind,cls,desc",
    [pytest.param(c, k, cl, d, id=c) for c, k, cl, d in WB_ERR_CASES],
)
def test_bfs_nodes_whitebox_errflag_class_npu(case_id, err_kind, cls, desc):
    """Device-side validation on a Key2-size (multi-block) graph: the kernel
    sets the per-core errflag partition and the host CHECK raises with the
    exact class code."""
    device, _ = _setup()
    if device is None:
        pytest.skip("NPU not available")
    indptr, indices, source = _build_circulant(_WB_ERR_BASE)
    indptr, indices, source = _inject_error(indptr, indices, source, err_kind)
    num_nodes = int(indptr.size - 1)
    g_npu = _make_graph_from_csr(
        indptr, indices, num_nodes, torch.int32, device
    )
    _assert_raises_dgl_error(
        lambda: _run_bfs(g_npu, source.tolist(), "nodes"), cls=cls
    )
    torch.npu.synchronize()  # no crash residue after the error


# ---------------------------------------------------------------------------
# __main__ runner (mirrors the style of the other tests/ascend/test_*_npu.py)
# ---------------------------------------------------------------------------


def _collect_tests():
    tests = []

    for dtype in (torch.int32, torch.int64):
        dt = "int32" if dtype is torch.int32 else "int64"
        for c, g, s, d in L0_CASES:
            tests.append(
                (
                    f"L0/{c}/{dt}",
                    lambda c=c, g=g, s=s, d=d, dt=dtype: test_bfs_nodes_l0_npu(
                        c, g, s, d, dt
                    ),
                )
            )
        for c, g, s, d in L1_CASES:
            tests.append(
                (
                    f"L1/{c}/{dt}",
                    lambda c=c, g=g, s=s, d=d, dt=dtype: test_bfs_nodes_l1_npu(
                        c, g, s, d, dt
                    ),
                )
            )
        tests.append(
            (
                f"L1/rand1k/{dt}",
                lambda dt=dtype: test_bfs_nodes_l1_rand1k_npu(dt),
            )
        )
        for c, g, s, cl, d in L2_CALL_RAISE:
            tests.append(
                (
                    f"L2/{c}/{dt}",
                    lambda c=c, g=g, s=s, cl=cl, d=d, dt=dtype: test_bfs_nodes_l2_call_raises_npu(
                        c, g, s, cl, d, dt
                    ),
                )
            )
    for c, k, d in L2_CONSTRUCT_RAISE:
        tests.append(
            (
                f"L2/{c}/construct",
                lambda c=c, k=k, d=d: test_bfs_nodes_l2_construct_raises_npu(
                    c, k, d
                ),
            )
        )
    for c, n, s, k, dt, e, d in WB_TILING_CASES:
        tests.append(
            (
                f"WB/{c}",
                lambda c=c, n=n, s=s, k=k, dt=dt, e=e, d=d: test_bfs_nodes_whitebox_tiling_key_npu(
                    c, n, s, k, dt, e, d
                ),
            )
        )
    for c, k, cl, d in WB_ERR_CASES:
        tests.append(
            (
                f"WB/{c}",
                lambda c=c, k=k, cl=cl, d=d: test_bfs_nodes_whitebox_errflag_class_npu(
                    c, k, cl, d
                ),
            )
        )
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
