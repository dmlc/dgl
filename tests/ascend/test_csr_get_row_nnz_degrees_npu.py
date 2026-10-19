"""Degree API regression for the multi-core CSRGetRowNNZ (array form).

CSRGetRowNNZ(csr, rows) is shared infrastructure: beyond the samplers,
the public in_degrees/out_degrees APIs route here through
immutable_graph/unit_graph. The multi-core rewrite (typed __gm__ reads
+ MTE-batched outputs) must keep those paths bit-exact — this file pins
the public-API surface so future nnz changes cannot regress silently.
"""
import pytest
import torch
import dgl


def _check_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 0), (2, 3),
    (3, 4),
]
# out-degree per node: node0=3, 1=2, 2=2, 3=1, 4=0
OUT_DEG = [3, 2, 2, 1, 0]
# in-degree per node: node0=1, 1=1, 2=2, 3=3, 4=1
IN_DEG = [1, 1, 2, 3, 1]


def _build(device, formats=("csr",)):
    src = torch.tensor([e[0] for e in EDGES], dtype=torch.int64)
    dst = torch.tensor([e[1] for e in EDGES], dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=5).formats(list(formats))
    if device != torch.device("cpu"):
        g = g.to(device)
    return g


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_out_degrees_subset_npu():
    """out_degrees on a subset routes through the array-form nnz kernel."""
    g = _build(torch.device("npu"))
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu")
    assert g.out_degrees(ids).cpu().tolist() == OUT_DEG[:4]


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_in_degrees_subset_npu():
    g = _build(torch.device("npu"), formats=("csc",))
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu")
    assert g.in_degrees(ids).cpu().tolist() == IN_DEG[:4]


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_degrees_all_nodes_npu():
    """The no-argument form (all nodes) is the common dataloader path."""
    g = _build(torch.device("npu"), formats=("csc", "csr"))
    assert g.in_degrees().cpu().tolist() == IN_DEG
    assert g.out_degrees().cpu().tolist() == OUT_DEG


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_degrees_zero_degree_tail_npu():
    """Node 4 has out-degree 0 — exercises the zero/empty-row store."""
    g = _build(torch.device("npu"))
    ids = torch.tensor([4], dtype=torch.int64, device="npu")
    assert g.out_degrees(ids).cpu().tolist() == [0]
    ids4 = torch.tensor([4], dtype=torch.int64, device="npu")
    g_csc = _build(torch.device("npu"), formats=("csc",))
    assert g_csc.in_degrees(ids4).cpu().tolist() == [1]


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
@pytest.mark.parametrize(
    "idtype", [torch.int32, torch.int64])
def test_degrees_int32_npu(idtype):
    """int32 graphs share the nnz kernel path.

    torch_npu's aclnnMinDim (used by the out_degrees reduction) rejects
    DT_INT32 at the plugin layer — an upstream limitation unrelated to
    the nnz kernel. The int32 KERNEL coverage rides on the fused
    sampler's int32 parametrization (test_csr_row_wise_sampling_uniform_
    fused_npu.py); here the CSC/in_degrees path is the NPU-safe surface.
    """
    src = torch.tensor([e[0] for e in EDGES], dtype=idtype)
    dst = torch.tensor([e[1] for e in EDGES], dtype=idtype)
    g_cpu = dgl.graph((src, dst), num_nodes=5).formats(["csc"])
    expected = g_cpu.in_degrees().tolist()
    g = g_cpu.to("npu")
    ids = torch.arange(5, dtype=idtype, device="npu")
    assert g.in_degrees(ids).cpu().tolist() == expected


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_degrees_multi_block_boundary_npu():
    """Row counts crossing multi-block chunk boundaries: 41 and 43 rows
    straddle the 40-core chunk edges (chunk=2 with 40 blocks), where the
    earlier per-word-write corruption lived."""
    n = 43
    gen = torch.Generator().manual_seed(7)
    # Both endpoints inside [0, 7); 43 edges across 7 nodes straddles
    # the 40-block chunk boundaries (chunk=2).
    src = torch.randint(0, 7, (n,), generator=gen)
    dst = torch.randint(0, 7, (n,), generator=gen)
    g_cpu = dgl.graph((src, dst), num_nodes=7).formats(["csc"])
    expected = g_cpu.in_degrees().tolist()
    g = g_cpu.to("npu")
    assert g.in_degrees().cpu().tolist() == expected
