"""
Test COOToCSR conversion on Ascend NPU.

Verifies that CSRMatrix produced from COO on NPU matches the CPU reference.
Covers sorted/unsorted COO, with/without data, int32/int64, and edge cases.
"""
import sys
import pytest
import torch
import dgl
from dgl.sparse import from_coo


def _check_npu_available():
    return hasattr(torch, 'npu') and torch.npu.is_available()


TEST_CASES = [
    ("sorted_basic",       [0, 0, 1, 2], [1, 2, 0, 3], (3, 4), True),
    ("unsorted_basic",     [1, 0, 2, 0], [0, 1, 3, 2], (3, 4), False),
    ("single_row",         [0, 0, 0],     [0, 1, 2],    (1, 3), True),
    ("same_row",           [2, 2, 2],     [0, 1, 2],    (5, 3), True),
    ("empty_trailing",     [0, 1],        [0, 1],      (5, 5), True),
    ("unsorted_gaps",      [4, 0, 2, 0],  [1, 0, 3, 2], (5, 4), False),
    ("single_entry",       [0],           [0],         (1, 1), True),
    ("reverse_sorted",     [3, 2, 1, 0],  [0, 1, 2, 3], (4, 4), False),
]


def _run_coo_to_csr(rows, cols, vals, shape, device, cpu_device):
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), shape)
    indptr_npu, indices_npu, _ = mat_npu.csr()

    mat_cpu = from_coo(rows.to(cpu_device), cols.to(cpu_device), vals.to(cpu_device), shape)
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    return indptr_npu, indices_npu, indptr_cpu, indices_cpu


def _check(name, indptr_npu, indices_npu, indptr_cpu, indices_cpu, vals_npu=None, vals_cpu=None):
    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"{name}: indptr mismatch"
    assert torch.allclose(indices_npu.cpu(), indices_cpu), f"{name}: indices mismatch"
    if vals_npu is not None and vals_cpu is not None:
        assert torch.allclose(vals_npu.cpu(), vals_cpu.cpu()), f"{name}: values mismatch"


# ─── Setup ─────────────────────────────────────────────────────

def _setup():
    if not _check_npu_available():
        return None, None
    device = torch.device('npu:0')
    cpu = torch.device('cpu')
    return device, cpu


# ─── Basic tests ───────────────────────────────────────────────

def test_coo_to_csr_npu_basic():
    """Sorted COO with data, int64 indices."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    cols = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
    vals = torch.randn(4, 8)

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows, cols, vals, (3, 4), device, cpu)

    _check("basic", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def test_coo_to_csr_npu_with_data():
    """COO with non-trivial data values."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    cols = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
    vals = torch.tensor([10, 20, 30, 40], dtype=torch.float32)

    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), vals.to(cpu), (3, 4))

    assert torch.allclose(mat_npu.val.cpu(), mat_cpu.val), f"with_data: NPU={mat_npu.val.cpu().tolist()}, CPU={mat_cpu.val.tolist()}"


def test_coo_to_csr_npu_without_data():
    """COO without data array (C++ auto-creates Range(0, nnz))."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    cols = torch.tensor([1, 2, 0, 3], dtype=torch.int64)

    mat_npu = from_coo(rows.to(device), cols.to(device), shape=(3, 4))
    indptr_npu, indices_npu, data_npu = mat_npu.csr()

    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), shape=(3, 4))
    indptr_cpu, indices_cpu, data_cpu = mat_cpu.csr()

    _check("without_data", indptr_npu, indices_npu, indptr_cpu, indices_cpu,
           data_npu, data_cpu)


# ─── Edge-ID (data field) correctness ──────────────────────────
# These tests assert the CSR matrix `data` field (edge IDs) is correct after
# COOToCSR. Previously the suite only checked indptr/indices/vals, so a bug
# where the no-data branch forgot to gather the synthesized Range(0,nnz) by the
# sort permutation (leaving data=identity while rows/cols were reordered,
# corrupting the position→eid mapping) went undetected. That mapping is what
# weighted in-edge neighbor sampling relies on (prob[data[pos]]).

# Unsorted COO with nodes that have multiple in-edges → sort perm is
# non-identity, so a wrong data array is observable. (rows, cols, shape).
EID_CASES = [
    ("unsorted_multi_in",     [0, 0, 0, 1, 1], [1, 2, 3, 0, 2], (4, 4)),
    ("unsorted_reorder",      [2, 0, 1, 0],     [0, 1, 3, 2],    (3, 4)),
    ("reverse_all",           [3, 2, 1, 0],     [0, 1, 2, 3],    (4, 4)),
    ("shuffled_large",        [4, 0, 2, 0, 3, 1, 4, 2],
                              [1, 0, 3, 2, 0, 4, 2, 1], (5, 5)),
]


def _check_csr_data(name, rows, cols, shape, dtype, device, cpu):
    """Build COO→CSR on NPU and CPU; assert indptr, indices, AND data match."""
    rows_t = torch.tensor(rows, dtype=dtype)
    cols_t = torch.tensor(cols, dtype=dtype)
    mat_npu = from_coo(rows_t.to(device), cols_t.to(device), shape=shape)
    mat_cpu = from_coo(rows_t.to(cpu), cols_t.to(cpu), shape=shape)
    indptr_n, indices_n, data_n = mat_npu.csr()
    indptr_c, indices_c, data_c = mat_cpu.csr()
    assert torch.equal(indptr_n.cpu(), indptr_c), f"{name}: indptr mismatch"
    assert torch.equal(indices_n.cpu(), indices_c), f"{name}: indices mismatch"
    assert torch.equal(data_n.cpu(), data_c), \
        f"{name}: data(eid) mismatch\n  npu={data_n.cpu().tolist()}\n  cpu={data_c.tolist()}"


@pytest.mark.parametrize("name,rows,cols,shape", [pytest.param(*c, id=c[0]) for c in EID_CASES])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_coo_to_csr_npu_data_field(name, rows, cols, shape, dtype):
    """CSR `data` (edge-ID) field must match CPU after COOToCSR reordering."""
    device, cpu = _setup()
    if device is None:
        return
    _check_csr_data(name, rows, cols, shape, dtype, device, cpu)


def test_coo_to_csr_npu_in_edges_eid():
    """Graph-level: in_edges(form='eid') must return the same edge IDs as CPU.

    This is the user-facing path that weighted in-edge sampling depends on.
    Exercises the CSC matrix (in-CSR) data array, built via COOToCSR.
    """
    device, cpu = _setup()
    if device is None:
        return

    # Unsorted COO; several nodes with multiple in-edges.
    src = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 3, 0, 2, 3], dtype=torch.int64)
    g_npu = dgl.graph((src.to(device), dst.to(device)))
    g_cpu = dgl.graph((src, dst))

    for n in range(g_cpu.num_nodes()):
        eid_cpu = g_cpu.in_edges(torch.tensor([n]), form="eid").tolist()
        eid_npu = g_npu.in_edges(torch.tensor([n], device=device), form="eid").cpu().tolist()
        assert eid_npu == eid_cpu, \
            f"in_edges({n}, eid): npu={eid_npu} cpu={eid_cpu}"

    # Also verify the full edge-id set round-trips through find_edges.
    eids = torch.arange(g_cpu.num_edges())
    s_n, d_n = g_npu.find_edges(eids.to(device))
    s_c, d_c = g_cpu.find_edges(eids)
    assert torch.equal(s_n.cpu(), s_c), "find_edges src mismatch"
    assert torch.equal(d_n.cpu(), d_c), "find_edges dst mismatch"


def test_coo_to_csr_npu_data_field_random_large():
    """Random unsorted COO: CSR data must match CPU (regression for the
    no-data-gather bug on large inputs)."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(0)
    num_rows, num_cols, nnz = 500, 300, 5000
    rows = torch.randint(0, num_rows, (nnz,))
    cols = torch.randint(0, num_cols, (nnz,))
    for dtype in (torch.int32, torch.int64):
        _check_csr_data(f"random_large_{dtype}", rows.tolist(), cols.tolist(),
                        (num_rows, num_cols), dtype, device, cpu)


# ─── int32 parametrized ────────────────────────────────────────

@pytest.mark.parametrize(
    "name,rows,cols,shape,row_sorted",
    [pytest.param(n, r, c, s, f, id=n) for n, r, c, s, f in TEST_CASES],
)
def test_coo_to_csr_npu_int32(name, rows, cols, shape, row_sorted):
    device, cpu = _setup()
    if device is None:
        return

    rows_t = torch.tensor(rows, dtype=torch.int32)
    cols_t = torch.tensor(cols, dtype=torch.int32)
    vals = torch.randn(len(rows))

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows_t, cols_t, vals, shape, device, cpu)

    _check(name, indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── int64 parametrized ────────────────────────────────────────

@pytest.mark.parametrize(
    "name,rows,cols,shape,row_sorted",
    [pytest.param(n, r, c, s, f, id=n) for n, r, c, s, f in TEST_CASES],
)
def test_coo_to_csr_npu_int64(name, rows, cols, shape, row_sorted):
    device, cpu = _setup()
    if device is None:
        return

    rows_t = torch.tensor(rows, dtype=torch.int64)
    cols_t = torch.tensor(cols, dtype=torch.int64)
    vals = torch.randn(len(rows))

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows_t, cols_t, vals, shape, device, cpu)

    _check(name, indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── Large random test ─────────────────────────────────────────

def test_coo_to_csr_npu_random_large():
    """Random COO with 10K entries, unsorted, int32."""
    device, cpu = _setup()
    if device is None:
        return

    num_rows, num_cols = 1000, 500
    nnz = 10000
    rows_t = torch.randint(0, num_rows, (nnz,), dtype=torch.int32)
    cols_t = torch.randint(0, num_cols, (nnz,), dtype=torch.int32)
    vals = torch.randn(nnz)

    indptr_npu, indices_npu, indptr_cpu, indices_cpu = \
        _run_coo_to_csr(rows_t, cols_t, vals, (num_rows, num_cols), device, cpu)

    _check("random_large", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── Graph path test ────────────────────────────────────────────

def test_coo_to_csr_npu_graph_path():
    """COOToCSR via dgl.graph path (internal COO->CSR conversion)."""
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 0])

    g_npu = dgl.graph((src.to(device), dst.to(device)))
    adj_npu = g_npu.adjacency_matrix()

    g_cpu = dgl.graph((src, dst))
    adj_cpu = g_cpu.adjacency_matrix()

    indptr_npu, indices_npu, _ = adj_npu.csr()
    indptr_cpu, indices_cpu, _ = adj_cpu.csr()

    _check("graph_path", indptr_npu, indices_npu, indptr_cpu, indices_cpu)
    assert g_npu.device == device


# ─── Unified-path anchor: sorted vs unsorted produce identical CSR ────────
# The counting-sort rewrite must produce bit-identical output for row_sorted
# and unsorted inputs of the same logical edge set (stable semantics). The
# unsorted case is the sorted case's edges permuted; CPU COOToCSR's stable
# sort guarantees equal rows keep their original relative order, so feeding
# the SAME order in both flags must give identical output. We feed the same
# edge order twice with only the row_sorted flag flipped: for an actually
# sorted input both runs are semantically identical, and for an unsorted
# input the flag is a caller assertion (garbage-in), so we test the
# meaningful direction: sorted-order edges must render identically whether
# the flag says sorted or not.

def test_coo_to_csr_npu_sorted_flag_invariance():
    """Same edge order, row_sorted flag flipped: output must be identical."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(7)
    num_rows, nnz = 200, 4000
    rows = torch.sort(torch.randint(0, num_rows, (nnz,)))[0]
    cols = torch.randint(0, num_rows, (nnz,))

    outs = []
    for dtype in (torch.int32, torch.int64):
        for flag in (True, False):
            # int32 graphs must be built on CPU then moved (aclnnMaxDim
            # rejects NPU int32 max at construction time).
            g = dgl.graph(
                (rows.to(dtype), cols.to(dtype)),
                num_nodes=num_rows, row_sorted=flag).to(device)
            adj = g.adjacency_matrix()
            indptr, indices, data = adj.csr()
            outs.append((indptr.cpu(), indices.cpu(), data.cpu()))
        assert torch.equal(outs[-2][0], outs[-1][0]), f"{dtype}: indptr differs by flag"
        assert torch.equal(outs[-2][1], outs[-1][1]), f"{dtype}: indices differ by flag"
        assert torch.equal(outs[-2][2], outs[-1][2]), f"{dtype}: data differs by flag"


# ─── Cache semantics ───────────────────────────────────────────────────────
# These tests fail on an implementation without a cache only in the sense of
# "not applicable"; they target the cache that the rewrite introduces:
#   1. repeated conversion on the same graph must return correct results
#      (the cache must not corrupt the second call);
#   2. tensor-address reuse (free old tensor, allocate a different-content
#      tensor that lands on the same address) must NOT hit the old entry.

def _graph_csr(graph):
    adj = graph.adjacency_matrix()
    return adj.csr()


def test_coo_to_csr_npu_repeat_call_consistency():
    """Converting the same graph repeatedly must stay correct (cache path)."""
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 1, 2, 0, 1, 3], dtype=torch.int64)
    dst = torch.tensor([1, 2, 3, 2, 0, 1], dtype=torch.int64)
    g_npu = dgl.graph((src.to(device), dst.to(device))).formats("coo")
    g_cpu = dgl.graph((src, dst)).formats("coo")

    for i in range(3):
        indptr_n, indices_n, data_n = _graph_csr(g_npu)
        indptr_c, indices_c, data_c = _graph_csr(g_cpu)
        assert torch.equal(indptr_n.cpu(), indptr_c), f"call {i}: indptr mismatch"
        assert torch.equal(indices_n.cpu(), indices_c), f"call {i}: indices mismatch"
        assert torch.equal(data_n.cpu(), data_c), f"call {i}: data mismatch"


def test_coo_to_csr_npu_address_reuse_no_stale_hit():
    """Free a converted COO, build a different one that reuses the device
    memory: the conversion result must reflect the NEW content, not a stale
    cache entry keyed on the recycled pointer."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(1)
    n, m = 64, 512
    src = torch.randint(0, n, (m,), dtype=torch.int64)
    dst = torch.randint(0, n, (m,), dtype=torch.int64)

    g1 = dgl.graph((src.to(device), dst.to(device))).formats("coo")
    indptr1, indices1, _ = _graph_csr(g1)
    del g1  # release the graph; its COO arrays are freed

    # Different content, same shapes and dtype: the NPU caching allocator is
    # likely to hand back the same addresses. A pointer-keyed cache without
    # input retention would return the old CSR here.
    src2 = (src + 1) % n
    g2 = dgl.graph((src2.to(device), dst.to(device))).formats("coo")
    indptr2, indices2, _ = _graph_csr(g2)

    g2_cpu = dgl.graph((src2, dst)).formats("coo")
    indptr_c, indices_c, _ = _graph_csr(g2_cpu)

    assert torch.equal(indptr2.cpu(), indptr_c), \
        "stale cache hit on address reuse: indptr is from the OLD graph"
    assert torch.equal(indices2.cpu(), indices_c), \
        "stale cache hit on address reuse: indices is from the OLD graph"


def test_coo_to_csr_npu_graph_content_change_same_object():
    """A graph whose COO content differs must never reuse another graph's
    CSR even when shapes match (distinct tensors → distinct keys)."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(2)
    n, m = 50, 300
    a_src = torch.randint(0, n, (m,), dtype=torch.int64)
    a_dst = torch.randint(0, n, (m,), dtype=torch.int64)
    b_src = torch.randint(0, n, (m,), dtype=torch.int64)
    b_dst = torch.randint(0, n, (m,), dtype=torch.int64)

    g_a = dgl.graph((a_src.to(device), a_dst.to(device))).formats("coo")
    g_b = dgl.graph((b_src.to(device), b_dst.to(device))).formats("coo")

    indptr_a, indices_a, _ = _graph_csr(g_a)
    indptr_b, indices_b, _ = _graph_csr(g_b)

    cpu_a = dgl.graph((a_src, a_dst)).formats("coo")
    cpu_b = dgl.graph((b_src, b_dst)).formats("coo")
    indptr_ca, indices_ca, _ = _graph_csr(cpu_a)
    indptr_cb, indices_cb, _ = _graph_csr(cpu_b)

    assert torch.equal(indptr_a.cpu(), indptr_ca), "graph A indptr wrong"
    assert torch.equal(indptr_b.cpu(), indptr_cb), "graph B indptr wrong"
    assert torch.equal(indices_a.cpu(), indices_ca), "graph A indices wrong"
    assert torch.equal(indices_b.cpu(), indices_cb), "graph B indices wrong"
    # And the two graphs must not collide with each other.
    assert not torch.equal(indptr_a.cpu(), indptr_b.cpu()) or \
        not torch.equal(indices_a.cpu(), indices_b.cpu()), \
        "distinct graphs collided to the same CSR (cache key collision)"


# ─── Defense: out-of-range row ids must fail loudly ────────────────────────

def test_coo_to_csr_npu_row_out_of_range():
    """A row id >= num_rows must trigger an error (kernel-side clamping is
    NOT acceptable; host-side CHECK from the min/max reduction is)."""
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 1, 999], dtype=torch.int64)  # 999 >= num_rows=3
    cols = torch.tensor([0, 1, 2], dtype=torch.int64)
    with pytest.raises(Exception):
        dgl.graph((rows.to(device), cols.to(device)), num_nodes=3).formats("csr")


# ─── Large-scale smoke: multi-band path (rows beyond single-band capacity) ─

def test_coo_to_csr_npu_multi_band_large():
    """num_rows large enough that the counting-sort bands split: correctness
    against CPU on a random sparse matrix."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(3)
    # 2M rows / 500k edges: rows span far beyond a 192KB band (~48k int32
    # slots), forcing multiple bands in Pass 1/2.
    num_rows, nnz = 2_000_000, 500_000
    rows = torch.randint(0, num_rows, (nnz,), dtype=torch.int64)
    cols = torch.randint(0, 1000, (nnz,), dtype=torch.int64)

    g_npu = dgl.graph((rows.to(device), cols.to(device)),
                      num_nodes=num_rows).formats("coo")
    g_cpu = dgl.graph((rows, cols), num_nodes=num_rows).formats("coo")

    indptr_n, indices_n, data_n = _graph_csr(g_npu)
    indptr_c, indices_c, data_c = _graph_csr(g_cpu)

    assert torch.equal(indptr_n.cpu(), indptr_c), "multi-band indptr mismatch"
    assert torch.equal(indices_n.cpu(), indices_c), "multi-band indices mismatch"
    assert torch.equal(data_n.cpu(), data_c), "multi-band data mismatch"


# ─── Sampling integration (the DEBT-01 user story) ─────────────────────────

def test_coo_to_csr_npu_sampling_integration():
    """COO-restricted graph sampling (the 0.43x scenario) must stay correct
    while the conversion is cached/recomputed."""
    device, cpu = _setup()
    if device is None:
        return

    torch.manual_seed(4)
    n, m = 200, 2000
    src = torch.randint(0, n, (m,), dtype=torch.int64)
    dst = torch.randint(0, n, (m,), dtype=torch.int64)
    g_npu = dgl.graph((src.to(device), dst.to(device)),
                      num_nodes=n).formats("coo")
    g_cpu = dgl.graph((src, dst), num_nodes=n).formats("coo")

    nodes = torch.arange(n)
    for fanout in (10, -1):
        sg_npu = dgl.sampling.sample_neighbors(
            g_npu, nodes.to(device), fanout, edge_dir="out")
        sg_cpu = dgl.sampling.sample_neighbors(
            g_cpu, nodes, fanout, edge_dir="out")
        # Sampled subgraphs come back as CSR (eid order); compare per-SRC
        # sampled degrees (bincount over the SOURCE tensor — counting dst
        # measures in-degree pattern and mismatches randomly): both sides
        # must pick exactly min(fanout, out_deg) edges per source node.
        sn, _ = sg_npu.cpu().edges()
        sc, _ = sg_cpu.edges()
        npu_deg = torch.bincount(sn.long(), minlength=n)
        cpu_deg = torch.bincount(sc.long(), minlength=n)
        out_deg = torch.bincount(src, minlength=n)
        cap = n if fanout == -1 else fanout  # select-all: no cap
        expect = torch.minimum(out_deg, torch.full_like(out_deg, cap))
        assert torch.equal(npu_deg, expect), \
            f"fanout={fanout}: NPU per-source degree != min(fanout, out_deg)"
        assert torch.equal(cpu_deg, expect), \
            f"fanout={fanout}: CPU per-source degree != min(fanout, out_deg)"
        # select-all: total sampled edges must equal total edges
        if fanout == -1:
            assert sn.numel() == m, f"select-all: {sn.numel()} != {m} edges"


if __name__ == "__main__":
    failures = 0
    named_tests = [
        ("basic", test_coo_to_csr_npu_basic),
        ("with_data", test_coo_to_csr_npu_with_data),
        ("without_data", test_coo_to_csr_npu_without_data),
    ]
    for name, rows, cols, shape, row_sorted in TEST_CASES:
        named_tests.append(
            (f"{name}_int32", lambda n=name, r=rows, c=cols, s=shape, f=row_sorted:
             test_coo_to_csr_npu_int32(n, r, c, s, f)))
        named_tests.append(
            (f"{name}_int64", lambda n=name, r=rows, c=cols, s=shape, f=row_sorted:
             test_coo_to_csr_npu_int64(n, r, c, s, f)))
    named_tests.append(("random_large", test_coo_to_csr_npu_random_large))
    named_tests.append(("graph_path", test_coo_to_csr_npu_graph_path))
    # Edge-ID (data field) coverage — would have caught the no-data-gather bug.
    for name, rows, cols, shape in EID_CASES:
        for dtype, dname in ((torch.int32, "int32"), (torch.int64, "int64")):
            named_tests.append((
                f"data_field_{name}_{dname}",
                lambda n=name, r=rows, c=cols, s=shape, dt=dtype:
                test_coo_to_csr_npu_data_field(n, r, c, s, dt)))
    named_tests.append(("in_edges_eid", test_coo_to_csr_npu_in_edges_eid))
    named_tests.append(("data_field_random_large", test_coo_to_csr_npu_data_field_random_large))
    named_tests.append(("sorted_flag_invariance", test_coo_to_csr_npu_sorted_flag_invariance))
    named_tests.append(("repeat_call_consistency", test_coo_to_csr_npu_repeat_call_consistency))
    named_tests.append(("address_reuse_no_stale_hit", test_coo_to_csr_npu_address_reuse_no_stale_hit))
    named_tests.append(("graph_content_change_same_object", test_coo_to_csr_npu_graph_content_change_same_object))
    named_tests.append(("row_out_of_range", test_coo_to_csr_npu_row_out_of_range))
    named_tests.append(("multi_band_large", test_coo_to_csr_npu_multi_band_large))
    named_tests.append(("sampling_integration", test_coo_to_csr_npu_sampling_integration))
    for name, test in named_tests:
        try:
            test()
            print(f"  PASS [{name}]")
        except AssertionError as e:
            print(f"  FAIL [{name}] {e}")
            failures += 1
    total = len(named_tests)
    print(f"\nResults: {total - failures}/{total} passed, {failures} failed", flush=True)

