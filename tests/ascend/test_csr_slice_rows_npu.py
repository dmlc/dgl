"""
Test csr_slice_rows (CSRSliceRows) on Ascend NPU.

Exercises the Ascend CSRSliceRows kernel: extracts a subset of rows from a
CSR matrix, returning a new CSR with renumbered rows. Supports both
continuous range and arbitrary row-id-array selection.
"""
import torch
import dgl


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check(name, npu_val, cpu_val):
    assert torch.allclose(npu_val.cpu(), cpu_val), f"{name}: max_diff={(npu_val.cpu() - cpu_val).abs().max().item():.6e}"


def test_slice_rows_first_n():
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    dst = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    rows = torch.tensor([0, 1])
    sub_npu = g_npu.subgraph(rows.to(device))
    sub_cpu = g_cpu.subgraph(rows)
    _check("first_n_edges_src", sub_npu.edges()[0], sub_cpu.edges()[0])
    _check("first_n_edges_dst", sub_npu.edges()[1], sub_cpu.edges()[1])


def test_slice_rows_skip():
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    dst = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    rows = torch.tensor([0, 2, 3])
    sub_npu = g_npu.subgraph(rows.to(device))
    sub_cpu = g_cpu.subgraph(rows)
    _check("skip_edges_src", sub_npu.edges()[0], sub_cpu.edges()[0])
    _check("skip_edges_dst", sub_npu.edges()[1], sub_cpu.edges()[1])


def test_slice_single_row():
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    rows = torch.tensor([1])
    sub_npu = g_npu.subgraph(rows.to(device))
    sub_cpu = g_cpu.subgraph(rows)
    assert sub_npu.num_nodes() == 1
    assert sub_cpu.num_nodes() == 1
    _check("single_row_edges_src", sub_npu.edges()[0], sub_cpu.edges()[0])


def test_slice_all_rows():
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    rows = torch.tensor([0, 1, 2, 3])
    sub_npu = g_npu.subgraph(rows.to(device))
    sub_cpu = g_cpu.subgraph(rows)
    _check("all_rows_edges_src", sub_npu.edges()[0], sub_cpu.edges()[0])
    _check("all_rows_edges_dst", sub_npu.edges()[1], sub_cpu.edges()[1])


def test_slice_empty_rows():
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    rows = torch.tensor([], dtype=torch.long)
    sub_npu = g_npu.subgraph(rows.to(device))
    sub_cpu = g_cpu.subgraph(rows)
    assert sub_npu.num_nodes() == 0
    assert sub_cpu.num_nodes() == 0


def test_slice_reverse_order():
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    dst = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    rows = torch.tensor([3, 1, 0])
    sub_npu = g_npu.subgraph(rows.to(device))
    sub_cpu = g_cpu.subgraph(rows)
    _check("reverse_edges_src", sub_npu.edges()[0], sub_cpu.edges()[0])
    _check("reverse_edges_dst", sub_npu.edges()[1], sub_cpu.edges()[1])


if __name__ == "__main__":
    tests = [test_slice_rows_first_n, test_slice_rows_skip, test_slice_single_row, test_slice_all_rows, test_slice_empty_rows, test_slice_reverse_order]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

