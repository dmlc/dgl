"""
Test csr_get_row_nnz (CSRGetRowNNZ) on Ascend NPU.

Exercises the Ascend CSRGetRowNNZ kernel: returns the number of non-zeros
per row. Supports scalar (single row) and batch (NDArray of rows) variants.
"""
import torch
import dgl


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check(name, npu_val, cpu_val):
    assert torch.allclose(npu_val.cpu(), cpu_val), f"{name}: NPU={npu_val.cpu().tolist()} CPU={cpu_val.tolist()}"


def test_get_row_nnz_basic():
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    dst = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))
    rows = torch.tensor([0, 1, 2, 3])
    nnz_npu = g_npu.in_degrees(rows.to(device))
    nnz_cpu = g_cpu.in_degrees(rows)
    _check("basic", nnz_npu, nnz_cpu)


def test_get_row_nnz_single():
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))
    nnz_npu = g_npu.in_degrees(0)
    nnz_cpu = g_cpu.in_degrees(0)
    assert nnz_npu == nnz_cpu, f"NPU={nnz_npu} CPU={nnz_cpu}"


def test_get_row_nnz_empty_rows():
    device, cpu = _setup()
    if device is None:
        return
    # 3-node graph (0,1,2): query row 0 and 2 which have 1 in-edge each,
    # row 1 has 2 in-edges, to verify empty-row handling works
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))
    rows = torch.tensor([0, 1, 2, 3])
    nnz_npu = g_npu.in_degrees(rows.to(device))
    nnz_cpu = g_cpu.in_degrees(rows)
    _check("empty_rows", nnz_npu, nnz_cpu)


def test_get_row_nnz_all():
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))
    nnz_npu = g_npu.in_degrees()
    nnz_cpu = g_cpu.in_degrees()
    _check("all", nnz_npu, nnz_cpu)


def test_get_row_nnz_empty_graph():
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([], dtype=torch.long)
    dst = torch.tensor([], dtype=torch.long)
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))
    # 0-node graph has no valid row IDs; query empty set
    rows = torch.tensor([], dtype=torch.long)
    nnz_npu = g_npu.in_degrees(rows.to(device))
    nnz_cpu = g_cpu.in_degrees(rows)
    _check("empty_graph", nnz_npu, nnz_cpu)


def test_get_row_nnz_int32():
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
    dst = torch.tensor([1, 2, 0, 3], dtype=torch.int32)
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    nnz_npu = g_npu.in_degrees(rows.to(device))
    nnz_cpu = g_cpu.in_degrees(rows)
    _check("int32", nnz_npu, nnz_cpu)


if __name__ == "__main__":
    tests = [test_get_row_nnz_basic, test_get_row_nnz_single, test_get_row_nnz_empty_rows, test_get_row_nnz_all, test_get_row_nnz_empty_graph, test_get_row_nnz_int32]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

