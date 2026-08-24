"""
Test suite for PR #29: _npu_gspmm_binary (NPU-native SpMM binary op for mul/add)

Covers the NPU-native fallback added to _gspmm() in _sparse_ops.py.
When device is NPU and op in ("mul","add") and reduce_op=="sum",
_gspmm dispatches to _npu_gspmm_binary instead of _CAPI_DGLKernelSpMM.

Test strategy:
  1. Correctness: NPU output vs CPU reference for op="mul"/"add", 2D/1D features
  2. Gradient: backward pass through _gspmm on NPU vs CPU
  3. DGL integration: update_all(fn.u_mul_e, fn.sum) end-to-end
  4. Edge cases: single edge, self-loops, large feature dim, non-intercepted ops
"""
import pytest
import numpy as np
import torch
import dgl
import dgl.function as fn
from dgl._sparse_ops import _gspmm

try:
    import torch_npu  # noqa: F401
    has_npu = True
except ImportError:
    has_npu = False


def get_device():
    if has_npu and torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


@pytest.fixture
def small_graph():
    g = dgl.graph((
        torch.tensor([0, 1, 1, 2, 2, 3, 0, 4]),
        torch.tensor([1, 0, 2, 1, 3, 2, 4, 0]),
    ), num_nodes=5)
    return g.int()


@pytest.fixture
def medium_graph():
    num_nodes = 20
    edges = 60
    src = torch.randint(0, num_nodes, (edges,))
    dst = torch.randint(0, num_nodes, (edges,))
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g.int()


def run_gspmm_cpu_npu(g, node_feats, edge_feats, op, reduce_op, device):
    """Run _gspmm on CPU (reference) and NPU, return both outputs."""
    v_cpu, _ = _gspmm(g._graph, op, reduce_op,
                      node_feats.clone(), edge_feats.clone())

    g_dev = g.to(device)
    v_dev, _ = _gspmm(g_dev._graph, op, reduce_op,
                      node_feats.to(device), edge_feats.to(device))
    v_dev_cpu = v_dev.cpu()

    return v_cpu, v_dev_cpu


def assert_close(a, b, atol=1e-5, rtol=1e-4):
    np.testing.assert_allclose(
        a.numpy(), b.numpy(), atol=atol, rtol=rtol,
        err_msg=f"Mismatch: max_diff={torch.abs(a - b).max().item()}")


# ============================================================================
# 1. Correctness tests: NPU output vs CPU reference
# ============================================================================

@pytest.mark.skipif(not has_npu, reason="NPU not available")
class TestGspmmBinaryCorrectness:
    """Test correctness of _npu_gspmm_binary vs CPU reference."""

    def test_mul_sum_2d_small(self, small_graph):
        """op=mul, reduce=sum, 2D features, small graph (5 nodes, 8 edges)."""
        g = small_graph
        u = torch.randn(g.num_nodes(), 4)
        e = torch.randn(g.num_edges(), 4)
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "mul", "sum", get_device())
        assert_close(v_cpu, v_npu)

    def test_add_sum_2d_small(self, small_graph):
        """op=add, reduce=sum, 2D features, small graph."""
        g = small_graph
        u = torch.randn(g.num_nodes(), 4)
        e = torch.randn(g.num_edges(), 4)
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "add", "sum", get_device())
        assert_close(v_cpu, v_npu)

    def test_mul_sum_2d_medium(self, medium_graph):
        """op=mul, reduce=sum, 2D features, medium graph (20 nodes, 60 edges)."""
        g = medium_graph
        u = torch.randn(g.num_nodes(), 8)
        e = torch.randn(g.num_edges(), 8)
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "mul", "sum", get_device())
        assert_close(v_cpu, v_npu)

    def test_add_sum_2d_medium(self, medium_graph):
        """op=add, reduce=sum, 2D features, medium graph."""
        g = medium_graph
        u = torch.randn(g.num_nodes(), 8)
        e = torch.randn(g.num_edges(), 8)
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "add", "sum", get_device())
        assert_close(v_cpu, v_npu)

    def test_mul_sum_feat_dim_64(self, medium_graph):
        """op=mul, reduce=sum, feature dim=64."""
        g = medium_graph
        u = torch.randn(g.num_nodes(), 64)
        e = torch.randn(g.num_edges(), 64)
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "mul", "sum", get_device())
        assert_close(v_cpu, v_npu)

    def test_mul_sum_1d_features(self, small_graph):
        """op=mul, reduce=sum, 1D (scalar per node) features."""
        g = small_graph
        u = torch.randn(g.num_nodes())
        e = torch.randn(g.num_edges())
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "mul", "sum", get_device())
        assert_close(v_cpu, v_npu)

    def test_add_sum_1d_features(self, small_graph):
        """op=add, reduce=sum, 1D (scalar per node) features."""
        g = small_graph
        u = torch.randn(g.num_nodes())
        e = torch.randn(g.num_edges())
        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "add", "sum", get_device())
        assert_close(v_cpu, v_npu)


# ============================================================================
# 2. Gradient tests: backward through _gspmm on NPU
# ============================================================================

@pytest.mark.skipif(not has_npu, reason="NPU not available")
class TestGspmmBinaryGradient:
    """Test backward through update_all(fn.u_mul_e, fn.sum) on NPU.
    Since _gspmm is a low-level C API call without autograd graph,
    we test gradients via DGL's update_all which wraps GSpMM autograd.
    """

    def test_backward_u_mul_e_sum(self, small_graph):
        """Backward of update_all(u_mul_e, sum): compare grad on NPU vs CPU."""
        torch.manual_seed(42)
        g = small_graph
        device = get_device()

        u_cpu = torch.randn(g.num_nodes(), 4, requires_grad=True)
        e_cpu = torch.randn(g.num_edges(), 4, requires_grad=True)

        with g.local_scope():
            g.ndata["h"] = u_cpu
            g.edata["w"] = e_cpu
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "out"))
            loss_cpu = g.ndata["out"].sum()
        loss_cpu.backward()

        g_dev = g.to(device)
        u_dev = u_cpu.detach().to(device).requires_grad_(True)
        e_dev = e_cpu.detach().to(device).requires_grad_(True)
        with g_dev.local_scope():
            g_dev.ndata["h"] = u_dev
            g_dev.edata["w"] = e_dev
            g_dev.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "out"))
            loss_dev = g_dev.ndata["out"].sum()
        loss_dev.backward()

        assert_close(u_cpu.grad, u_dev.grad.cpu(), atol=1e-4)
        assert_close(e_cpu.grad, e_dev.grad.cpu(), atol=1e-4)

    def test_backward_u_add_e_sum(self, small_graph):
        """Backward of update_all(u_add_e, sum): compare grad on NPU vs CPU."""
        torch.manual_seed(42)
        g = small_graph
        device = get_device()

        u_cpu = torch.randn(g.num_nodes(), 4, requires_grad=True)
        e_cpu = torch.randn(g.num_edges(), 4, requires_grad=True)

        with g.local_scope():
            g.ndata["h"] = u_cpu
            g.edata["w"] = e_cpu
            g.update_all(fn.u_add_e("h", "w", "m"), fn.sum("m", "out"))
            loss_cpu = g.ndata["out"].sum()
        loss_cpu.backward()

        g_dev = g.to(device)
        u_dev = u_cpu.detach().to(device).requires_grad_(True)
        e_dev = e_cpu.detach().to(device).requires_grad_(True)
        with g_dev.local_scope():
            g_dev.ndata["h"] = u_dev
            g_dev.edata["w"] = e_dev
            g_dev.update_all(fn.u_add_e("h", "w", "m"), fn.sum("m", "out"))
            loss_dev = g_dev.ndata["out"].sum()
        loss_dev.backward()

        assert_close(u_cpu.grad, u_dev.grad.cpu(), atol=1e-4)
        assert_close(e_cpu.grad, e_dev.grad.cpu(), atol=1e-4)

    def test_backward_mul_sum_medium_l2(self, medium_graph):
        """Backward with L2 loss on medium graph, feature dim=16."""
        torch.manual_seed(123)
        g = medium_graph
        device = get_device()

        u_cpu = torch.randn(g.num_nodes(), 16, requires_grad=True)
        e_cpu = torch.randn(g.num_edges(), 16, requires_grad=True)

        with g.local_scope():
            g.ndata["h"] = u_cpu
            g.edata["w"] = e_cpu
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "out"))
            loss_cpu = (g.ndata["out"] ** 2).sum()
        loss_cpu.backward()

        g_dev = g.to(device)
        u_dev = u_cpu.detach().to(device).requires_grad_(True)
        e_dev = e_cpu.detach().to(device).requires_grad_(True)
        with g_dev.local_scope():
            g_dev.ndata["h"] = u_dev
            g_dev.edata["w"] = e_dev
            g_dev.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "out"))
            loss_dev = (g_dev.ndata["out"] ** 2).sum()
        loss_dev.backward()

        assert_close(u_cpu.grad, u_dev.grad.cpu(), atol=1e-3)
        assert_close(e_cpu.grad, e_dev.grad.cpu(), atol=1e-3)


# ============================================================================
# 3. DGL integration tests: update_all(fn.u_mul_e, fn.sum)
# ============================================================================

@pytest.mark.skipif(not has_npu, reason="NPU not available")
class TestGspmmBinaryIntegration:
    """Test through DGL's update_all API with fn.u_mul_e + fn.sum."""

    def test_update_all_u_mul_e_sum(self, small_graph):
        """End-to-end update_all(fn.u_mul_e, fn.sum) on NPU vs CPU."""
        g = small_graph
        device = get_device()
        u = torch.randn(g.num_nodes(), 4)
        e = torch.randn(g.num_edges(), 4)

        with g.local_scope():
            g.ndata["h"] = u
            g.edata["w"] = e
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "out"))
            out_cpu = g.ndata["out"].clone()

        g_dev = g.to(device)
        with g_dev.local_scope():
            g_dev.ndata["h"] = u.to(device)
            g_dev.edata["w"] = e.to(device)
            g_dev.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "out"))
            out_npu = g_dev.ndata["out"].cpu()

        assert_close(out_cpu, out_npu)

    def test_update_all_u_add_e_sum(self, small_graph):
        """End-to-end update_all(fn.u_add_e, fn.sum) on NPU vs CPU."""
        g = small_graph
        device = get_device()
        u = torch.randn(g.num_nodes(), 4)
        e = torch.randn(g.num_edges(), 4)

        with g.local_scope():
            g.ndata["h"] = u
            g.edata["w"] = e
            g.update_all(fn.u_add_e("h", "w", "m"), fn.sum("m", "out"))
            out_cpu = g.ndata["out"].clone()

        g_dev = g.to(device)
        with g_dev.local_scope():
            g_dev.ndata["h"] = u.to(device)
            g_dev.edata["w"] = e.to(device)
            g_dev.update_all(fn.u_add_e("h", "w", "m"), fn.sum("m", "out"))
            out_npu = g_dev.ndata["out"].cpu()

        assert_close(out_cpu, out_npu)


# ============================================================================
# 4. Edge case and non-intercepted path tests
# ============================================================================

@pytest.mark.skipif(not has_npu, reason="NPU not available")
class TestGspmmBinaryEdgeCases:
    """Test edge cases and verify non-intercepted paths still work."""

    def test_copy_lhs_sum_not_intercepted(self, small_graph):
        """op=copy_lhs should use original SpMM kernel, not _npu_gspmm_binary."""
        g = small_graph
        device = get_device()
        u = torch.randn(g.num_nodes(), 4)

        v_cpu, _ = _gspmm(g._graph, "copy_lhs", "sum", u, None)

        g_dev = g.to(device)
        v_dev, _ = _gspmm(g_dev._graph, "copy_lhs", "sum", u.to(device), None)
        assert_close(v_cpu, v_dev.cpu())

    def test_single_edge_graph(self):
        """Graph with a single edge."""
        g = dgl.graph(([0], [1]), num_nodes=2).int()
        device = get_device()
        u = torch.randn(2, 4)
        e = torch.randn(1, 4)

        v_cpu, _ = _gspmm(g._graph, "mul", "sum", u, e)

        g_dev = g.to(device)
        v_dev, _ = _gspmm(g_dev._graph, "mul", "sum",
                          u.to(device), e.to(device))
        assert_close(v_cpu, v_dev.cpu())

    def test_self_loop_graph(self):
        """Graph with self-loops."""
        g = dgl.graph(([0, 1, 2, 0, 1], [0, 1, 2, 1, 0]),
                      num_nodes=3).int()
        device = get_device()
        u = torch.randn(3, 4)
        e = torch.randn(5, 4)

        v_cpu, _ = _gspmm(g._graph, "mul", "sum", u, e)

        g_dev = g.to(device)
        v_dev, _ = _gspmm(g_dev._graph, "mul", "sum",
                          u.to(device), e.to(device))
        assert_close(v_cpu, v_dev.cpu())

    def test_large_feature_dim(self, medium_graph):
        """Feature dimension of 128."""
        g = medium_graph
        u = torch.randn(g.num_nodes(), 128)
        e = torch.randn(g.num_edges(), 128)

        v_cpu, v_npu = run_gspmm_cpu_npu(g, u, e, "mul", "sum", get_device())
        assert_close(v_cpu, v_npu, atol=1e-4)
