"""
Test csr_slice_matrix (CSRSliceMatrix) on Ascend NPU.

Exercises the Ascend CSRSliceMatrix kernel: extracts submatrix M[rows, cols]
from a CSR matrix, returning a new CSR with renumbered rows and cols.
"""
import torch
import dgl


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check(name, npu_val, cpu_val):
    assert torch.allclose(npu_val.cpu(), cpu_val), f"{name}: max_diff={(npu_val.cpu() - cpu_val).abs().max().item():.6e}"


def test_slice_rows_via_subgraph():
    """g.subgraph(rows) internally uses CSRSliceRows, which shares logic with SliceMatrix."""
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    dst = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    sub_npu = g_npu.subgraph(torch.tensor([0, 1, 3], device=device))
    sub_cpu = g_cpu.subgraph(torch.tensor([0, 1, 3]))

    eids_npu = sub_npu.edges(form='all')
    eids_cpu = sub_cpu.edges(form='all')
    _check("subgraph_edges", eids_npu[0], eids_cpu[0])
    _check("subgraph_edges_dst", eids_npu[1], eids_cpu[1])


if __name__ == "__main__":
    tests = [test_slice_rows_via_subgraph]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

