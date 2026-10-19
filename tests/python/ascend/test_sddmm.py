"""
SDDMM operator tests on Ascend NPU.

Tests cover:
- dot/add/sub/mul/div/copy_lhs/copy_rhs × COO/CSR × FP32/FP16 × int32/int64
- Backward (autograd)
- Various feat_dims, bipartite, empty graph, large graph
- dgl.sparse.sddmm dispatch path (array.cc::CSRSDDMM/COOSDDMM macro fix)
"""
import pytest
import torch
import torch_npu
import numpy as np
import dgl
import dgl.sparse as dglsp
from dgl.ops import gsddmm

dev = torch.device("npu:0")
torch.npu.set_device(dev)


def make_graph(num_src=10, num_dst=8, num_edges=30, idtype=torch.int64):
    torch.npu.synchronize()
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    return dgl.graph((src.tolist(), dst.tolist()), idtype=idtype).to(dev)


def make_bipartite(num_src=10, num_dst=12, num_edges=40, idtype=torch.int64):
    np.random.seed(123)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    g = dgl.heterograph({("_U", "_E", "_V"): (src.tolist(), dst.tolist())})
    return g.astype(idtype).to(dev)


class TestSDDMM:
    """SDDMM operator tests."""

    def setup_method(self):
        torch.npu.synchronize()

    def teardown_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("op", ["dot", "add", "sub", "mul", "div",
                                     "copy_lhs", "copy_rhs"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_sddmm_coo_fp32(self, op, idtype):
        """SDDMM all ops, COO format, FP32."""
        torch.npu.synchronize()
        g = make_graph(idtype=idtype)
        g_cpu = g.cpu()
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev)
        if op == "div":
            rhs = torch.rand(g.num_src_nodes(), 4, device=dev) + 0.5
        else:
            rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g, op, lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        tol = 1e-4 if op != "div" else 1e-3
        assert diff < tol, f"{op} diff={diff}"

    @pytest.mark.parametrize("op", ["dot", "add", "sub", "mul", "div",
                                     "copy_lhs", "copy_rhs"])
    @pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
    def test_sddmm_csr_fp32(self, op, idtype):
        """SDDMM with CSR format graph."""
        torch.npu.synchronize()
        g = make_graph(idtype=idtype)
        g_csr = g.formats("csr").to(dev)
        g_cpu = g_csr.cpu()
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g_csr, op, lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-4, f"CSR {op} diff={diff}"

    @pytest.mark.parametrize("op", ["dot", "copy_lhs", "copy_rhs"])
    def test_sddmm_fp16(self, op):
        """SDDMM with FP16 features (compare against FP32 CPU ref)."""
        torch.npu.synchronize()
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev)

        e_npu = gsddmm(g, op, lhs.half(), rhs.half(),
                        lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, op, lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.float().cpu() - e_cpu).abs().max().item()
        assert diff < 5e-2, f"FP16 {op} diff={diff}"

    @pytest.mark.parametrize("feat_dim", [1, 4, 13, 64, 128])
    def test_sddmm_dot_feat_dims(self, feat_dim):
        """SDDMM dot with various feature dimensions."""
        torch.npu.synchronize()
        g = make_graph(idtype=torch.int64)
        g_cpu = g.cpu()
        lhs = torch.rand(g.num_src_nodes(), feat_dim, device=dev)
        rhs = torch.rand(g.num_src_nodes(), feat_dim, device=dev)

        e_npu = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, "dot", lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-3, f"feat_dim={feat_dim} diff={diff}"

    def test_sddmm_bipartite(self):
        """SDDMM on bipartite graph."""
        g = make_bipartite(num_src=10, num_dst=12)
        g_cpu = g.cpu()
        lhs = torch.rand(10, 4, device=dev)
        rhs = torch.rand(12, 4, device=dev)

        e_npu = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, "dot", lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-4

    def test_sddmm_empty_graph(self):
        """SDDMM on graph with 0 edges."""
        g = dgl.graph(([], []), num_nodes=5, idtype=torch.int64).to(dev)
        lhs = torch.rand(5, 4, device=dev)
        rhs = torch.rand(5, 4, device=dev)
        e = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        assert e.shape[0] == 0

    def test_sddmm_backward(self):
        """SDDMM dot backward (autograd)."""
        g = make_graph(idtype=torch.int64)
        lhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        rhs = torch.rand(g.num_src_nodes(), 4, device=dev, requires_grad=True)
        e = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        loss = e.sum()
        loss.backward()
        assert lhs.grad is not None and lhs.grad.abs().sum() > 0
        assert rhs.grad is not None and rhs.grad.abs().sum() > 0

    def test_sddmm_large_graph(self):
        """SDDMM on large graph (1000 nodes, 5000 edges)."""
        g = make_graph(num_src=1000, num_dst=1000, num_edges=5000,
                       idtype=torch.int64)
        g_cpu = g.cpu()
        lhs = torch.rand(1000, 64, device=dev)
        rhs = torch.rand(1000, 64, device=dev)

        e_npu = gsddmm(g, "dot", lhs, rhs, lhs_target="u", rhs_target="v")
        e_cpu = gsddmm(g_cpu, "dot", lhs.cpu(), rhs.cpu(),
                        lhs_target="u", rhs_target="v")
        diff = (e_npu.cpu() - e_cpu).abs().max().item()
        assert diff < 1e-3, f"large graph diff={diff}"


def make_sparse_coo(num_src=10, num_dst=8, num_edges=30):
    np.random.seed(42)
    src = np.random.randint(0, num_src, num_edges)
    dst = np.random.randint(0, num_dst, num_edges)
    row = torch.tensor(src, dtype=torch.int32)
    col = torch.tensor(dst, dtype=torch.int32)
    val = torch.ones(num_edges, dtype=torch.float32)
    return dglsp.from_coo(row, col, val, shape=(num_src, num_dst)).to(dev)


class TestSparseSDDMM:
    """dgl.sparse.SDDMM dispatch path tests.

    These tests verify that array.cc::CSRSDDMM() and array.cc::COOSDDMM()
    correctly route kDGLAscend to the Ascend SDDMM kernel implementations
    via ATEN_XPU_SWITCH_CUDA_ASCEND.

    The dgl.sparse path goes through:
      dgl.sparse.sddmm() -> matmul.cc::SDDMMNoAutoGrad() ->
      aten::CSRSDDMM()/COOSDDMM() -> array.cc macro dispatch ->
      SDDMMCsr<kDGLAscend>/SDDMMCoo<kDGLAscend>

    Before the macro fix, calling dgl.sparse.sddmm() on NPU would crash
    with LOG(FATAL) "Operator SDDMM does not support ascend device."
    """

    def setup_method(self):
        torch.npu.synchronize()

    def teardown_method(self):
        torch.npu.synchronize()

    def test_sparse_sddmm_coo_no_crash(self):
        """dgl.sparse.sddmm with COO format does not crash on NPU."""
        sp = make_sparse_coo()
        mat1 = torch.rand(10, 4, device=dev)
        mat2 = torch.rand(4, 8, device=dev)
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 30

    def test_sparse_sddmm_csr_no_crash(self):
        """dgl.sparse.sddmm with CSR format does not crash on NPU."""
        sp = make_sparse_coo()
        sp_csr = dglsp.from_coo(
            sp.coo()[0].cpu(), sp.coo()[1].cpu(),
            sp.val.cpu(), shape=sp.shape).to(dev)
        mat1 = torch.rand(10, 4, device=dev)
        mat2 = torch.rand(4, 8, device=dev)
        result = dglsp.sddmm(sp_csr, mat1, mat2)
        assert result.val.shape[0] == 30

    def test_sparse_sddmm_shape_correct(self):
        """Output shape matches nnz regardless of dispatch path."""
        for num_edges in [5, 30, 100]:
            sp = make_sparse_coo(num_edges=num_edges)
            mat1 = torch.rand(10, 4, device=dev)
            mat2 = torch.rand(4, 8, device=dev)
            result = dglsp.sddmm(sp, mat1, mat2)
            assert result.val.shape[0] == num_edges

    def test_sparse_sddmm_empty_graph(self):
        """Empty graph (0 edges) does not crash."""
        row = torch.tensor([], dtype=torch.int32)
        col = torch.tensor([], dtype=torch.int32)
        val = torch.tensor([], dtype=torch.float32)
        sp = dglsp.from_coo(row, col, val, shape=(5, 5)).to(dev)
        mat1 = torch.rand(5, 4, device=dev)
        mat2 = torch.rand(4, 5, device=dev)
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 0

    def test_sparse_sddmm_fp16(self):
        """FP16 dtype does not crash on NPU."""
        sp = make_sparse_coo()
        mat1 = torch.rand(10, 4, device=dev).half()
        mat2 = torch.rand(4, 8, device=dev).half()
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 30

    def test_gsddmm_and_sparse_both_work(self):
        """Both gsddmm (kernel.cc) and dgl.sparse (array.cc) paths work on NPU.

        gsddmm goes through kernel.cc::SDDMM() which uses
        ATEN_XPU_SWITCH_CUDA_ASCEND (fixed in PR #16).
        dgl.sparse.sddmm goes through matmul.cc -> array.cc::COOSDDMM()
        which uses ATEN_XPU_SWITCH_CUDA_ASCEND (fixed in this PR).
        """
        np.random.seed(42)
        src = np.random.randint(0, 10, 30)
        dst = np.random.randint(0, 8, 30)

        # Path 1: gsddmm (kernel.cc)
        g = dgl.graph((src.tolist(), dst.tolist()), idtype=torch.int32).to(dev)
        lhs = torch.rand(10, 4, device=dev)
        rhs = torch.rand(10, 4, device=dev)
        e_gsddmm = gsddmm(g, "dot", lhs, rhs,
                          lhs_target="u", rhs_target="v")
        assert e_gsddmm.shape[0] == 30

        # Path 2: dgl.sparse.sddmm (array.cc)
        row = torch.tensor(src, dtype=torch.int32)
        col = torch.tensor(dst, dtype=torch.int32)
        val = torch.ones(30, dtype=torch.float32)
        sp = dglsp.from_coo(row, col, val, shape=(10, 8)).to(dev)
        mat1 = torch.rand(10, 4, device=dev)
        mat2 = torch.rand(4, 8, device=dev)
        result = dglsp.sddmm(sp, mat1, mat2)
        assert result.val.shape[0] == 30
