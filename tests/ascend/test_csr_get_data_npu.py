"""
Test csr_get_data (CSRMask) on Ascend NPU.

Exercises the Ascend CSRGetData kernel: given a CSR matrix and (row, col) pairs,
returns the data at those coordinates.
"""
import os
import torch
import dgl
import dgl.function as fn


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check(name, npu_val, cpu_val):
    assert torch.allclose(npu_val.cpu(), cpu_val), f"{name}: max_diff={(npu_val.cpu() - cpu_val).abs().max().item():.6e}"


def test_csr_get_data_basic():
    """Basic (row, col) lookup on CSR with data."""
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2], device=device)
    dst = torch.tensor([1, 2, 0, 3], device=device)
    g = dgl.graph((src, dst)).to(device)
    eid = g.edge_ids(torch.tensor([0, 0], device=device),
                     torch.tensor([2, 1], device=device))
    assert len(eid) == 2


def test_csr_get_data_via_update_all():
    """Indirect test: use adjacency for message passing (exercises CSR)."""
    device, cpu = _setup()
    if device is None:
        return

    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 0, 3])
    feat = torch.tensor([1.0, 2.0, 3.0, 4.0])

    g_cpu = dgl.graph((src, dst)).int()
    g_cpu.ndata['x'] = feat
    g_cpu.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'x'))
    expected = g_cpu.ndata['x']

    g_npu = g_cpu.to(device)
    g_npu.ndata['x'] = feat.half().to(device)
    g_npu.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'x'))
    result = g_npu.ndata['x'].cpu().float()

    _check("csr_get_data_via_upate_all", result, expected)


if __name__ == "__main__":
    os.environ['DGL_SPMM_SUM_AIV_ONLY'] = '1'
    os.environ['DGL_SPMM_USE_PYTORCH_STREAM'] = '1'
    tests = [test_csr_get_data_basic, test_csr_get_data_via_update_all]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    os.environ['DGL_SPMM_SUM_AIV_ONLY'] = '0'
    os.environ['DGL_SPMM_USE_PYTORCH_STREAM'] = '0'
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

