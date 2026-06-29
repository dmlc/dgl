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
    indptr_npu, indices_npu, _ = mat_npu.csr()

    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), shape=(3, 4))
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    _check("without_data", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


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
    for name, test in named_tests:
        try:
            test()
            print(f"  PASS [{name}]")
        except AssertionError as e:
            print(f"  FAIL [{name}] {e}")
            failures += 1
    total = len(named_tests)
    print(f"\nResults: {total - failures}/{total} passed, {failures} failed", flush=True)

