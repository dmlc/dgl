"""
SPMM operator tests on Ascend NPU.

Tests cover:
- gspmm copy_lhs + sum/max/min × FP32/FP16 × int32/int64
- update_all(copy_u, sum/max/min) fused path
- Backward (autograd)
- Various feat_dims, large graph, zero-degree, empty graph, bipartite
- FP32 bit-exact precision, FP32 vs FP16 consistency
- Stream sync: in_edges/in_degrees after update_all/gsddmm
- Repeated calls stability
- segment_reduce (mean_nodes)
- End-to-end: GCN forward/backward, multi-layer, training loop
"""
import pytest
import torch
import torch_npu
import numpy as np
import dgl
import dgl.function as fn
from dgl.ops import gsddmm, gspmm

dev = torch.device("npu:0")
torch.npu.set_device(dev)


def make_graph(num_src=10, num_dst=10, num_edges=30, idtype=torch.int64):
    torch.npu.synchronize()
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    return dgl.graph((src.tolist(), dst.tolist()), idtype=idtype).to(dev)


class TestSPMM:
    """SPMM operator tests."""

    def setup_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_gspmm_fp32(self, reducer, idtype):
        """gspmm copy_lhs + sum/max/min, FP32."""
        g = make_graph(idtype=idtype)
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", reducer, ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", reducer, ufeat.cpu(), None)
        if reducer in ["max", "min"]:
            v_npu = dgl.backend.replace_inf_with_zero(v_npu)
            v_cpu = dgl.backend.replace_inf_with_zero(v_cpu)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4, f"{reducer} FP32 diff={diff}"

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_gspmm_fp16(self, reducer, idtype):
        """gspmm copy_lhs + sum/max/min, FP16 (compare against FP32 CPU ref)."""
        g = make_graph(idtype=idtype)
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", reducer, ufeat.half(), None)
        v_cpu = gspmm(g_cpu, "copy_lhs", reducer, ufeat.cpu(), None)
        if reducer in ["max", "min"]:
            v_npu = dgl.backend.replace_inf_with_zero(v_npu)
            v_cpu = dgl.backend.replace_inf_with_zero(v_cpu)
        diff = (v_npu.float().cpu() - v_cpu).abs().max().item()
        tol = 5.0 if reducer == "sum" else 5e-2
        assert diff < tol, f"{reducer} FP16 diff={diff}"

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    def test_update_all_fp32(self, reducer):
        """update_all(copy_u, sum/max/min) fused SpMM path, FP32."""
        g = make_graph()
        g_cpu = g.cpu()
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev)
        g_cpu.ndata["x"] = g.ndata["x"].cpu()

        reduce_fn = getattr(fn, reducer)("m", "h")
        g.update_all(fn.copy_u("x", "m"), reduce_fn)
        g_cpu.update_all(fn.copy_u("x", "m"), reduce_fn)
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4, f"update_all {reducer} diff={diff}"

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    def test_update_all_fp16(self, reducer):
        """update_all(copy_u, sum/max/min) fused SpMM path, FP16 (vs FP32 CPU)."""
        g = make_graph()
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_nodes(), 4, device=dev)
        g.ndata["x"] = ufeat.half()
        g_cpu.ndata["x"] = ufeat.cpu()

        reduce_fn = getattr(fn, reducer)("m", "h")
        g.update_all(fn.copy_u("x", "m"), reduce_fn)
        g_cpu.update_all(fn.copy_u("x", "m"), reduce_fn)
        diff = (g.ndata["h"].float().cpu() - g_cpu.ndata["h"]).abs().max().item()
        tol = 5.0 if reducer == "sum" else 5e-2
        assert diff < tol, f"update_all FP16 {reducer} diff={diff}"

    def test_backward_fp32(self):
        """SpMM FP32 forward + backward."""
        g = make_graph()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        v = gspmm(g, "copy_lhs", "sum", ufeat, None)
        loss = v.sum()
        loss.backward()
        assert ufeat.grad is not None

    def test_backward_fp16(self):
        """SpMM FP16 forward + backward."""
        g = make_graph()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        v = gspmm(g, "copy_lhs", "sum", ufeat.half(), None)
        v.sum().backward()
        assert ufeat.grad is not None

    @pytest.mark.parametrize("feat_dim", [1, 4, 13, 64, 128, 256])
    def test_feat_dims_fp32(self, feat_dim):
        """Various feature dimensions."""
        g = make_graph()
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), feat_dim, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"feat_dim={feat_dim} diff={diff}"

    def test_large_graph(self):
        """Large graph (1000 nodes, 5000 edges)."""
        g = make_graph(num_src=1000, num_dst=1000, num_edges=5000)
        g_cpu = g.cpu()
        ufeat = torch.rand(1000, 64, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-3, f"large graph diff={diff}"

    def test_zero_degree_nodes(self):
        """Zero-degree destination nodes should have zero output."""
        g = dgl.graph(([1, 2, 3], [1, 2, 3]), idtype=torch.int64).to(dev)
        g_cpu = g.cpu()
        ufeat = torch.rand(4, 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4
        assert v_npu[0].abs().max() == 0

    def test_empty_graph(self):
        """Graph with 0 edges."""
        g = dgl.graph(([], []), num_nodes=5, idtype=torch.int64).to(dev)
        ufeat = torch.rand(5, 4, device=dev)
        v = gspmm(g, "copy_lhs", "sum", ufeat, None)
        assert v.shape[0] == 5
        assert v.abs().max() == 0

    def test_int64_index(self):
        """int64 index (default graph type)."""
        g = dgl.graph(([0, 1, 2], [1, 2, 0])).to(dev)
        g_cpu = g.cpu()
        ufeat = torch.rand(3, 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4

    def test_bipartite(self):
        """Bipartite graph."""
        np.random.seed(123)
        src = np.random.randint(0, 10, 40)
        dst = np.random.randint(0, 12, 40)
        g = dgl.heterograph({("_U", "_E", "_V"): (src.tolist(), dst.tolist())})
        g = g.astype(torch.int64).to(dev)
        g_cpu = g.cpu()
        ufeat = torch.rand(10, 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        diff = (v_npu.cpu() - v_cpu).abs().max().item()
        assert diff < 1e-4

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    def test_repeated_calls(self, reducer):
        """Repeated calls on same graph (stability)."""
        g = make_graph()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev)
        results = []
        for _ in range(5):
            v = gspmm(g, "copy_lhs", reducer, ufeat, None)
            results.append(v.cpu())
        for i in range(1, 5):
            diff = (results[i] - results[0]).abs().max().item()
            assert diff == 0, f"Call {i} differs from call 0 by {diff}"

    def test_fp32_precision_bit_exact(self):
        """FP32 sum should be bit-exact."""
        g = make_graph()
        g_cpu = g.cpu()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev)

        v_npu = gspmm(g, "copy_lhs", "sum", ufeat, None)
        v_cpu = gspmm(g_cpu, "copy_lhs", "sum", ufeat.cpu(), None)
        assert torch.equal(v_npu.cpu(), v_cpu), "FP32 sum should be bit-exact"

    @pytest.mark.parametrize("reducer", ["sum", "max", "min"])
    def test_fp32_vs_fp16_consistency(self, reducer):
        """FP16 result should be close to FP32 result."""
        g = make_graph()
        ufeat = torch.rand(g.num_src_nodes(), 4, device=dev)

        v_fp32 = gspmm(g, "copy_lhs", reducer, ufeat, None)
        v_fp16 = gspmm(g, "copy_lhs", reducer, ufeat.half(), None)
        if reducer in ["max", "min"]:
            v_fp32 = dgl.backend.replace_inf_with_zero(v_fp32)
            v_fp16 = dgl.backend.replace_inf_with_zero(v_fp16)
        diff = (v_fp32.cpu() - v_fp16.float().cpu()).abs().max().item()
        tol = 5.0 if reducer == "sum" else 0.1
        assert diff < tol, f"FP32 vs FP16 {reducer} diff={diff}"


class TestStreamSync:
    """Stream synchronization and UDF reduce tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def test_udf_reduce_forward(self):
        """UDF reduce with degree bucketing."""
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev)
        g_cpu.ndata["x"] = g.ndata["x"].cpu()

        g.update_all(fn.copy_u("x", "m"),
                      lambda n: {"h": n.mailbox["m"].sum(1)})
        g_cpu.update_all(fn.copy_u("x", "m"),
                          lambda n: {"h": n.mailbox["m"].sum(1)})
        diff = (g.ndata["h"].cpu() - g_cpu.ndata["h"]).abs().max().item()
        assert diff < 1e-4

    def test_udf_reduce_backward(self):
        """UDF reduce forward + backward."""
        g = make_graph(idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev, requires_grad=True)
        g.update_all(fn.copy_u("x", "m"),
                      lambda n: {"h": n.mailbox["m"].sum(1)})
        g.ndata["h"].sum().backward()
        assert g.ndata["x"].grad is not None

    def test_in_edges_after_update_all(self):
        """in_edges correct after update_all."""
        g = make_graph(num_src=20, num_dst=20, num_edges=100, idtype=torch.int64)
        g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev, requires_grad=True)
        g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))

        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()
        assert g.in_degrees().sum().item() == g.num_edges()

    def test_in_edges_after_gsddmm(self):
        """in_edges correct after gsddmm."""
        g = make_graph(num_src=20, num_dst=20, num_edges=100, idtype=torch.int64)
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")

        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

    def test_repeated_update_all(self):
        """Multiple update_all calls (stability)."""
        g = make_graph(num_src=20, num_dst=20, num_edges=100, idtype=torch.int64)
        for _ in range(3):
            g.ndata["x"] = torch.rand(g.num_nodes(), 4, device=dev, requires_grad=True)
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            g.ndata["h"].sum().backward()

        nodes = torch.arange(g.num_nodes(), device=dev)
        eid = g.in_edges(nodes, form="eid")
        assert len(eid) == g.num_edges()

    def test_segment_reduce(self):
        """segment_reduce forward + backward."""
        from dgl.ops import segment_reduce
        seglen = torch.tensor([3, 2, 1], device=dev)
        value = torch.rand(6, 4, device=dev, requires_grad=True)
        y = segment_reduce(seglen, value, reducer="sum")
        assert y.shape == (3, 4)
        y.sum().backward()
        assert value.grad is not None


class TestEndToEnd:
    """End-to-end GNN model tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def test_gcn_forward_backward(self):
        """GCN forward + backward."""
        g = make_graph(num_src=30, num_dst=30, num_edges=100, idtype=torch.int64)
        g.ndata["x"] = torch.rand(30, 16, device=dev, requires_grad=True)
        g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
        linear = torch.nn.Linear(16, 8).to(dev)
        out = linear(g.ndata["h"])
        out.sum().backward()
        assert g.ndata["x"].grad is not None

    def test_multi_layer_gcn(self):
        """Multi-layer GCN with autograd."""
        g = make_graph(num_src=30, num_dst=30, num_edges=100, idtype=torch.int64)
        layers = [torch.nn.Linear(16, 16).to(dev), torch.nn.Linear(16, 8).to(dev)]
        x = torch.rand(30, 16, device=dev, requires_grad=True)
        for layer in layers:
            g.ndata["x"] = x
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            x = torch.relu(layer(g.ndata["h"]))
        x.sum().backward()
        assert layers[0].weight.grad is not None

    def test_training_loop(self):
        """Multi-step training."""
        g = make_graph(num_src=30, num_dst=30, num_edges=100, idtype=torch.int64)
        linear = torch.nn.Linear(16, 8).to(dev)
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
        for _ in range(5):
            g.ndata["x"] = torch.rand(30, 16, device=dev, requires_grad=True)
            g.update_all(fn.copy_u("x", "m"), fn.sum("m", "h"))
            out = linear(g.ndata["h"])
            loss = out.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert g.in_degrees().sum().item() == g.num_edges()
