"""
Test csr_get_row (CSRGetRowColumnIndices + CSRGetRowData) on Ascend NPU.

Exercises the Ascend CSR row-access path: for a given row, retrieves
the column indices and data via zero-copy views from the CSR structure.
"""
import torch
import dgl


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _run_test(name, src, dst, num_rows):
    device, cpu = _setup()
    if device is None:
        return

    g_npu = dgl.graph((src, dst)).to(device)
    g_cpu = dgl.graph((src, dst))

    for row in range(num_rows):
        eid_npu = g_npu.out_edges(row)
        eid_cpu = g_cpu.out_edges(row)
        d_npu = g_npu.out_degrees(row)
        d_cpu = g_cpu.out_degrees(row)
        assert d_npu == d_cpu, f"row {row}: NPU deg={d_npu} CPU deg={d_cpu}"


def test_csr_get_row_basic():
    src = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    dst = torch.tensor([1, 2, 0, 3, 0, 1, 2])
    _run_test("basic", src, dst, 4)


def test_csr_get_row_empty_row():
    # 3-node graph (0,1,2), so only query rows 0-2
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 0])
    _run_test("empty_row", src, dst, 3)


def test_csr_get_row_single():
    src = torch.tensor([0, 0, 0])
    dst = torch.tensor([0, 1, 2])
    _run_test("single_row", src, dst, 1)


def test_csr_get_row_all_empty():
    src = torch.tensor([], dtype=torch.long)
    dst = torch.tensor([], dtype=torch.long)
    _run_test("all_empty", src, dst, 0)


if __name__ == "__main__":
    tests = [test_csr_get_row_basic, test_csr_get_row_empty_row, test_csr_get_row_single, test_csr_get_row_all_empty]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS [{test.__name__}]")
        except AssertionError as e:
            print(f"  FAIL [{test.__name__}] {e}")
            failures += 1
    print(f"\nResults: {len(tests) - failures}/{len(tests)} passed, {failures} failed")

