"""
Test index_select (IndexSelect) on Ascend NPU.

Exercises the Ascend IndexSelect kernel: gathers elements from an NDArray
by index, like NumPy's A[index].
"""
import torch
from dgl.sparse import from_coo


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check(name, npu_val, cpu_val):
    assert torch.allclose(npu_val.cpu(), cpu_val), f"{name}: max_diff={(npu_val.cpu() - cpu_val).abs().max().item():.6e}"


def test_index_select_dim0_contiguous():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.tensor([10, 20, 30, 40], dtype=torch.float32)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows, cols, vals, (3, 4))

    idx = torch.tensor([0, 1])
    sub_npu = mat_npu.index_select(0, idx.to(device))
    sub_cpu = mat_cpu.index_select(0, idx)
    _check("dim0_contiguous_row", sub_npu.coo()[0], sub_cpu.coo()[0])
    _check("dim0_contiguous_col", sub_npu.coo()[1], sub_cpu.coo()[1])


def test_index_select_dim0_skip():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    cols = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    vals = torch.randn(7)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (4, 4))
    mat_cpu = from_coo(rows, cols, vals, (4, 4))

    idx = torch.tensor([0, 2, 3])
    sub_npu = mat_npu.index_select(0, idx.to(device))
    sub_cpu = mat_cpu.index_select(0, idx)
    _check("dim0_skip_row", sub_npu.coo()[0], sub_cpu.coo()[0])
    _check("dim0_skip_col", sub_npu.coo()[1], sub_cpu.coo()[1])


def test_index_select_dim0_reverse():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.randn(4)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows, cols, vals, (3, 4))

    idx = torch.tensor([2, 0, 1])
    sub_npu = mat_npu.index_select(0, idx.to(device))
    sub_cpu = mat_cpu.index_select(0, idx)
    _check("dim0_reverse_row", sub_npu.coo()[0], sub_cpu.coo()[0])


def test_index_select_dim1_contiguous():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.randn(4)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows, cols, vals, (3, 4))

    idx = torch.tensor([0, 1])
    sub_npu = mat_npu.index_select(1, idx.to(device))
    sub_cpu = mat_cpu.index_select(1, idx)
    _check("dim1_contiguous_col", sub_npu.coo()[1], sub_cpu.coo()[1])


def test_index_select_dim1_skip():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.randn(4)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows, cols, vals, (3, 4))

    idx = torch.tensor([0, 2, 3])
    sub_npu = mat_npu.index_select(1, idx.to(device))
    sub_cpu = mat_cpu.index_select(1, idx)
    _check("dim1_skip_col", sub_npu.coo()[1], sub_cpu.coo()[1])


def test_index_select_single():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.randn(4)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows, cols, vals, (3, 4))

    idx = torch.tensor([1])
    sub_npu = mat_npu.index_select(0, idx.to(device))
    sub_cpu = mat_cpu.index_select(0, idx)
    assert sub_npu.shape == sub_cpu.shape
    _check("single_row", sub_npu.coo()[0], sub_cpu.coo()[0])


def test_index_select_all():
    device, cpu = _setup()
    if device is None:
        return

    rows = torch.tensor([0, 0, 1, 2])
    cols = torch.tensor([1, 2, 0, 3])
    vals = torch.randn(4)
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (3, 4))
    mat_cpu = from_coo(rows, cols, vals, (3, 4))

    idx = torch.tensor([0, 1, 2])
    sub_npu = mat_npu.index_select(0, idx.to(device))
    sub_cpu = mat_cpu.index_select(0, idx)
    _check("all_rows_row", sub_npu.coo()[0], sub_cpu.coo()[0])
    _check("all_rows_col", sub_npu.coo()[1], sub_cpu.coo()[1])


if __name__ == "__main__":
    tests = [test_index_select_dim0_contiguous, test_index_select_dim0_skip, test_index_select_dim0_reverse, test_index_select_dim1_contiguous, test_index_select_dim1_skip, test_index_select_single, test_index_select_all]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

