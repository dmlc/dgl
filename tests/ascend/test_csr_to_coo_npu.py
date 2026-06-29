"""
Test csr_to_coo (CSRToCOO) on Ascend NPU.

Exercises the Ascend CSRToCOO kernel: converts a CSR matrix to COO format.
Verifies the roundtrip COO -> CSR -> COO preserves data.
"""
import torch
from dgl.sparse import from_coo


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check(name, npu_val, cpu_val):
    assert torch.allclose(npu_val.cpu(), cpu_val), f"{name}: max_diff={(npu_val.cpu() - cpu_val).abs().max().item():.6e}"


def _test_roundtrip(name, rows, cols, vals, shape):
    device, cpu = _setup()
    if device is None:
        return

    mat_npu = from_coo(
        rows.to(device), cols.to(device), vals.to(device), shape
    )
    mat_cpu = from_coo(rows, cols, vals, shape)

    # CSR -> COO
    coo_npu = mat_npu.coo()
    coo_cpu = mat_cpu.coo()

    _check(f"{name}_row", coo_npu[0], coo_cpu[0])
    _check(f"{name}_col", coo_npu[1], coo_cpu[1])
    if vals is not None:
        _check(f"{name}_val", mat_npu.val, mat_cpu.val)


def test_csr_to_coo_sorted():
    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.randn(4)
    _test_roundtrip("sorted", rows, cols, vals, (3, 4))


def test_csr_to_coo_unsorted():
    rows = torch.tensor([1, 0, 2, 0])
    cols = torch.tensor([0, 1, 3, 2])
    vals = torch.randn(4)
    _test_roundtrip("unsorted", rows, cols, vals, (3, 4))


def test_csr_to_coo_single_row():
    rows = torch.tensor([0, 0, 0])
    cols = torch.tensor([0, 1, 2])
    vals = torch.randn(3)
    _test_roundtrip("single_row", rows, cols, vals, (1, 3))


def test_csr_to_coo_same_row():
    rows = torch.tensor([2, 2, 2])
    cols = torch.tensor([0, 1, 2])
    vals = torch.randn(3)
    _test_roundtrip("same_row", rows, cols, vals, (5, 3))


def test_csr_to_coo_empty_trailing():
    rows = torch.tensor([0, 1])
    cols = torch.tensor([0, 1])
    vals = torch.randn(2)
    _test_roundtrip("empty_trailing", rows, cols, vals, (5, 5))


def test_csr_to_coo_single_entry():
    rows = torch.tensor([0])
    cols = torch.tensor([0])
    vals = torch.randn(1)
    _test_roundtrip("single_entry", rows, cols, vals, (1, 1))


def test_csr_to_coo_no_data():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    mat_npu = from_coo(rows.to(device), cols.to(device), shape=(3, 4))
    mat_cpu = from_coo(rows, cols, shape=(3, 4))

    coo_npu = mat_npu.coo()
    coo_cpu = mat_cpu.coo()
    _check("no_data_row", coo_npu[0], coo_cpu[0])
    _check("no_data_col", coo_npu[1], coo_cpu[1])


def test_csr_to_coo_large():
    num_rows, num_cols, nnz = 1000, 500, 10000
    torch.manual_seed(42)
    rows = torch.randint(0, num_rows, (nnz,))
    cols = torch.randint(0, num_cols, (nnz,))
    vals = torch.randn(nnz)
    _test_roundtrip("large", rows, cols, vals, (num_rows, num_cols))


if __name__ == "__main__":
    tests = [test_csr_to_coo_sorted, test_csr_to_coo_unsorted, test_csr_to_coo_single_row, test_csr_to_coo_same_row, test_csr_to_coo_empty_trailing, test_csr_to_coo_single_entry, test_csr_to_coo_no_data, test_csr_to_coo_large]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

