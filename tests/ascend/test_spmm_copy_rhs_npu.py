"""
Tests for gspmm copy_rhs (copy_e_sum) on Ascend NPU.

This covers the fix for:
- coo2csr.cc: npu_sorted_data not gathered by perm
- spmm.cc: SpMMCsrAscend copy_rhs support via GatherByIndex + isCopyRhs flag
- spmm_unified_aiv_kernel.cpp: CopyInBatch sequential read for copy_rhs

Test strategy:
  1. Correctness: NPU copy_rhs vs CPU reference, various graph shapes
  2. Asymmetric graph (non-trivial edge reordering)
  3. Various feat_dims, idtypes, reducers
  4. Backward (autograd)
  5. DGL integration: update_all(copy_e, sum) end-to-end
  6. Edge cases: single edge, self-loops, empty graph, zero-degree nodes
  7. Repeated calls stability
"""
import pytest
import numpy as np
import torch
import dgl
import dgl.function as fn
from dgl.ops import gspmm

try:
    import torch_npu  # noqa: F401
    has_npu = torch.npu.is_available()
except (ImportError, AttributeError):
    has_npu = False

dev = torch.device("npu:0") if has_npu else torch.device("cpu")
if has_npu:
    torch.npu.set_device(dev)
    torch.npu.config.allow_internal_format = False


def make_graph(num_src=10, num_dst=10, num_edges=30, idtype=torch.int64, seed=42):
    torch.npu.synchronize()
    np.random.seed(seed)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    return dgl.graph((src.tolist(), dst.tolist()), idtype=idtype).to(dev)


def make_asymmetric_graph(idtype=torch.int64):
    """Asymmetric graph where edge reordering is non-trivial.

    Edges: 0->1, 0->2, 1->2, 2->0
    CSC (by dst): dst=0 gets edge 3, dst=1 gets edge 0, dst=2 gets edges 1,2
    So data array should be [3, 0, 1, 2] (not identity [0,1,2,3]).
    """
    g = dgl.graph(
        (torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 0])),
        num_nodes=3,
        idtype=idtype,
    ).to(dev)
    return g


class TestCopyRhsCorrectness:
    """copy_rhs (copy_e_sum) correctness: NPU vs CPU."""

    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_copy_rhs_fp32_basic(self, idtype):
        """copy_e_sum FP32, random graph."""
        g = make_graph(idtype=idtype)
        g_cpu = g.cpu()
        efeat = torch.rand(g.num_edges(), 4, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"copy_rhs FP32 diff={diff}"

    def test_copy_rhs_asymmetric_graph(self):
        """Asymmetric graph with non-trivial edge reordering.

        Edges: 0->1(efeat=1), 0->2(efeat=2), 1->2(efeat=3), 2->0(efeat=4)
        Expected: out[0]=4, out[1]=1, out[2]=2+3=5
        """
        g = make_asymmetric_graph()
        g_cpu = g.cpu()
        efeat = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"asymmetric copy_rhs diff={diff}"

        # Verify exact values
        expected = torch.tensor([[4.0], [1.0], [5.0]])
        assert torch.allclose(v_npu.cpu(), expected, atol=1e-4), \
            f"Expected {expected.tolist()}, got {v_npu.cpu().tolist()}"

    @pytest.mark.parametrize("feat_dim", [1, 4, 13, 64, 128, 256])
    def test_copy_rhs_feat_dims(self, feat_dim):
        """Various feature dimensions."""
        g = make_graph()
        g_cpu = g.cpu()
        efeat = torch.rand(g.num_edges(), feat_dim, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"feat_dim={feat_dim} diff={diff}"

    def test_copy_rhs_1d_features(self):
        """1D edge features (scalar per edge)."""
        g = make_graph()
        g_cpu = g.cpu()
        efeat = torch.rand(g.num_edges(), device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"1D copy_rhs diff={diff}"

    def test_copy_rhs_large_graph(self):
        """Large graph (1000 nodes, 5000 edges)."""
        g = make_graph(num_src=1000, num_dst=1000, num_edges=5000)
        g_cpu = g.cpu()
        efeat = torch.rand(5000, 64, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"large graph copy_rhs diff={diff}"

    @pytest.mark.parametrize("reducer", ["sum"])
    def test_copy_rhs_reducers(self, reducer):
        """copy_rhs with sum/max/min reducers (only sum supported on NPU)."""
        g = make_graph()
        g_cpu = g.cpu()
        efeat = torch.rand(g.num_edges(), 4, device=dev)

        v_npu = gspmm(g, "copy_rhs", reducer, None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", reducer, None, efeat.cpu())
        if reducer in ["max", "min"]:
            v_npu = dgl.backend.replace_inf_with_zero(v_npu)
            v_cpu = dgl.backend.replace_inf_with_zero(v_cpu)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        tol = 1e-4 if reducer == "sum" else 1e-2
        assert diff < tol, f"copy_rhs {reducer} diff={diff}"


class TestCopyRhsBackward:
    """Backward pass through copy_rhs."""

    def test_copy_rhs_backward_fp32(self):
        """copy_rhs FP32 forward + backward."""
        g = make_graph()
        efeat = torch.rand(g.num_edges(), 4, device=dev, requires_grad=True)
        v = gspmm(g, "copy_rhs", "sum", None, efeat)
        v.sum().backward()
        assert efeat.grad is not None
        assert efeat.grad.shape == efeat.shape

    def test_copy_rhs_backward_asymmetric(self):
        """Backward on asymmetric graph."""
        g = make_asymmetric_graph()
        efeat = torch.rand(4, 4, device=dev, requires_grad=True)
        v = gspmm(g, "copy_rhs", "sum", None, efeat)
        v.sum().backward()
        assert efeat.grad is not None


class TestCopyRhsIntegration:
    """DGL integration: update_all(copy_e, sum)."""

    def test_update_all_copy_e_sum(self):
        """update_all(fn.copy_e, fn.sum) end-to-end."""
        g = make_graph()
        g_cpu = g.cpu()
        g.edata["e"] = torch.rand(g.num_edges(), 4, device=dev)
        g_cpu.edata["e"] = g.edata["e"].cpu()

        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "h"))
        g_cpu.update_all(fn.copy_e("e", "m"), fn.sum("m", "h"))
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4, f"update_all copy_e sum diff={diff}"

    def test_update_all_copy_e_sum_asymmetric(self):
        """update_all on asymmetric graph."""
        g = make_asymmetric_graph()
        g_cpu = g.cpu()
        g.edata["e"] = torch.rand(4, 4, device=dev)
        g_cpu.edata["e"] = g.edata["e"].cpu()

        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "h"))
        g_cpu.update_all(fn.copy_e("e", "m"), fn.sum("m", "h"))
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4, f"update_all asymmetric diff={diff}"


class TestCopyRhsEdgeCases:
    """Edge cases for copy_rhs."""

    def test_single_edge(self):
        """Single edge graph."""
        g = dgl.graph(([0], [1]), num_nodes=2).to(dev)
        g_cpu = g.cpu()
        efeat = torch.tensor([[1.0, 2.0, 3.0]], device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"single edge diff={diff}"

    def test_self_loops(self):
        """Self-loop graph."""
        g = dgl.graph(([0, 1, 2], [0, 1, 2]), num_nodes=3).to(dev)
        g_cpu = g.cpu()
        efeat = torch.rand(3, 4, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"self-loop diff={diff}"

    def test_zero_degree_nodes(self):
        """Zero-degree destination nodes should have zero output."""
        g = dgl.graph(([1, 2], [1, 2]), num_nodes=3).to(dev)
        g_cpu = g.cpu()
        efeat = torch.rand(2, 4, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4
        assert v_npu[0].abs().max() == 0, "Zero-degree node should have zero output"

    def test_empty_graph(self):
        """Graph with 0 edges."""
        g = dgl.graph(([], []), num_nodes=5).to(dev)
        efeat = torch.rand(0, 4, device=dev)
        v = gspmm(g, "copy_rhs", "sum", None, efeat)
        assert v.shape[0] == 5
        assert v.abs().max() == 0

    def test_bipartite(self):
        """Bipartite graph."""
        np.random.seed(123)
        src = np.random.randint(0, 10, 40)
        dst = np.random.randint(0, 12, 40)
        g = dgl.heterograph({("_U", "_E", "_V"): (src.tolist(), dst.tolist())})
        g = g.astype(torch.int64).to(dev)
        g_cpu = g.cpu()
        efeat = torch.rand(40, 4, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"bipartite copy_rhs diff={diff}"


class TestCopyRhsStability:
    """Stability and consistency tests."""

    def test_repeated_calls(self):
        """Repeated calls produce identical results."""
        g = make_graph()
        efeat = torch.rand(g.num_edges(), 4, device=dev)
        results = []
        for _ in range(5):
            v = gspmm(g, "copy_rhs", "sum", None, efeat)
            results.append(v.cpu())
        for i in range(1, 5):
            diff = (results[i] - results[0]).abs().max().item()
            assert diff == 0, f"Call {i} differs from call 0 by {diff}"

    def test_copy_rhs_vs_copy_lhs_consistency(self):
        """copy_e_sum and copy_u_sum should give same result when
        edge features == source node features (identity case)."""
        g = make_graph(num_src=5, num_dst=5, num_edges=10)
        ufeat = torch.rand(5, 4, device=dev)
        # Create efeat = ufeat[src] for each edge
        src_nodes = g.edges()[0]
        efeat = ufeat[src_nodes]

        v_copy_u = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_copy_e = gspmm(g, "copy_rhs", "sum", None, efeat)
        diff = (v_copy_u.cpu() - v_copy_e.cpu()).abs().max().item()
        assert diff < 1e-4, f"copy_u vs copy_e consistency diff={diff}"

    def test_fp32_bit_exact(self):
        """FP32 copy_rhs sum should be bit-exact vs CPU."""
        g = make_graph()
        g_cpu = g.cpu()
        efeat = torch.rand(g.num_edges(), 4, device=dev)

        v_npu = gspmm(g, "copy_rhs", "sum", None, efeat)
        v_cpu = gspmm(g_cpu, "copy_rhs", "sum", None, efeat.cpu())
        assert torch.equal(v_npu.cpu(), v_cpu), \
            "FP32 copy_rhs sum should be bit-exact"


class TestCopyRhsCooToCsr:
    """Verify COOToCSR data array correctness (the root cause fix)."""

    def test_coo_to_csr_data_non_identity(self):
        """COOToCSR on unsorted COO should produce non-identity data array."""
        # Asymmetric graph: dst = [1, 2, 2, 0] (not sorted)
        g = make_asymmetric_graph()
        # Get CSC (InCSR) matrix - internally calls COOToCSR(COOTranspose(coo))
        # The data array should NOT be identity [0,1,2,3]
        # It should be a permutation like [3, 0, 1, 2]
        csc = g._graph.GetCSCMatrix(0) if hasattr(g._graph, 'GetCSCMatrix') else None
        if csc is None:
            pytest.skip("GetCSCMatrix not accessible from Python")

        import dgl.base
        assert not dgl.base.isNullArray(csc.data), "CSC data should not be null"

        data_cpu = csc.data.cpu()
        # Data should NOT be identity [0, 1, 2, 3]
        is_identity = torch.equal(data_cpu, torch.arange(len(data_cpu)))
        assert not is_identity, \
            f"CSC data should be a permutation (non-identity), got {data_cpu.tolist()}"

    def test_coo_to_csr_data_correctness(self):
        """COOToCSR data array correctly maps sorted edges to original edges."""
        # Graph: 0->1, 0->2, 1->2, 2->0
        # CSC (by dst): dst=0 gets edge 3, dst=1 gets edge 0, dst=2 gets edges 1,2
        # So sorted order = [3, 0, 1, 2], data = [3, 0, 1, 2]
        g = make_asymmetric_graph()
        csc = g._graph.GetCSCMatrix(0) if hasattr(g._graph, 'GetCSCMatrix') else None
        if csc is None:
            pytest.skip("GetCSCMatrix not accessible from Python")

        data_cpu = csc.data.cpu()
        # Check that data maps correctly
        efeat = torch.tensor([10.0, 20.0, 30.0, 40.0])  # edge 0=10, 1=20, 2=30, 3=40
        gathered = efeat[data_cpu.long()]
        # After gather: [efeat[3], efeat[0], efeat[1], efeat[2]] = [40, 10, 20, 30]
        expected = torch.tensor([40.0, 10.0, 20.0, 30.0])
        assert torch.equal(gathered, expected), \
            f"Gathered efeat should be {expected.tolist()}, got {gathered.tolist()}"
