from timeit import default_timer
import dgl
import backend as F
import dgl.function as fn
import time
import numpy as np
import unittest, pytest
from test_utils import parametrize_dtype, get_cases

iters = 5
n_edge_scale = 1
num_rel_scale = 1

@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')
@unittest.skipIf(F._default_context_str == 'cpu', reason="Not implemented.")

@parametrize_dtype
def test_gathermm(idtype):
    def _test(feat_scale):
        in_feat = 16 * feat_scale
        out_feat = 8 * feat_scale
        print("in/out feat", in_feat, out_feat)
        E_per_rel = F.copy_to(F.tensor([50, 100, 20, 284, 89, 10, 82, 9200, 10, 20, 30, 100,
            128, 20, 284, 89, 10, 82, 92, 10, 20, 30, 100, 1280, 20, 284, 89, 1000, 82,
            92, 10, 2000, 30, 100, 128, 20, 284, 89, 10, 82, 92, 10, 20, 30]), F.cpu())

        E_per_rel *= n_edge_scale
        num_rel = len(E_per_rel)
        print('num_rel', num_rel)
        W_per_len = F.copy_to(F.full((num_rel,) ,in_feat, dtype=F.dtype(E_per_rel)), F.cpu())

        H_arr = []
        W_arr = []
        Out_arr = []
        Out_grad_arr = []

        for eid in range(num_rel):
            H_arr.append(F.randn((E_per_rel[eid], in_feat)))
            W_arr.append(F.randn((in_feat, out_feat)))
            Out_arr.append(F.zeros((E_per_rel[eid], out_feat)))
            Out_grad_arr.append(F.ones((E_per_rel[eid], out_feat)))

        H = F.cat([h for h in H_arr], 0)
        W = F.cat([w for w in W_arr], 0)
        W_3D = W.reshape(num_rel, in_feat, out_feat)
        Out = F.cat([out for out in Out_arr], 0)
        Out_grad = F.cat([o for o in Out_grad_arr], 0)

        print('H.shape', H.shape)
        print('W.shape', W.shape)
        print('W_3D.shape', W_3D.shape)
        print('Out.shape', Out.shape)

        etype_arr = []
        for eid in range(num_rel):
            etype_arr.append(F.full((E_per_rel[eid],), eid, dtype=F.dtype(E_per_rel)))
        etypes = F.cat([etype for etype in etype_arr], 0)

        #################################################################
        #  low-mem version using PyTorch operator
        #################################################################

        # forward pass
        out = []
        for i in range(len(E_per_rel)):
            Hi = H_arr[i]
            Wi = W_arr[i]
            out.append(F.matmul(Hi, Wi))
        out_low_mem = F.cat(out, 0)

        # backward pass
        H_grad = []
        W_grad = []
        for i in range(len(E_per_rel)):
            Hi = H_arr[i]
            Wi = W_arr[i]
            Out_gradi = Out_grad_arr[i]
            H_grad.append(F.matmul(Out_gradi, Wi.transpose(0,1)))
            W_grad.append(F.matmul(Hi.transpose(0,1), Out_gradi))
        Hgrad_low_mem = F.cat(H_grad, 0)
        Wgrad_low_mem = F.cat(W_grad, 0)
        Wgrad_low_mem = Wgrad_low_mem.reshape(num_rel, in_feat, out_feat)

        #################################################################
        #  gather_mm where H sorted according to etype
        #################################################################

        seglen_A = E_per_rel
        F.attach_grad(H)
        F.attach_grad(W_3D)
        with F.record_grad():
            out_gmm_sorted = dgl.ops.segment_mm(H, W_3D, seglen_A)
            F.backward(F.reduce_sum(out_gmm_sorted))
            Hgrad_gmm_sorted = H.grad
            Wgrad_gmm_sorted = W_3D.grad

        #################################################################
        #  gather_mm where H is not sorted (backward not supported yet)
        #################################################################

        F.attach_grad(H)
        F.attach_grad(W_3D)
        with F.record_grad():
            out_gmm_unsorted = dgl.ops.gather_mm(H, W_3D, idx_rhs=etypes)
            F.backward(F.reduce_sum(out_gmm_unsorted))
            Hgrad_gmm_unsorted = H.grad
            Wgrad_gmm_unsorted = W_3D.grad


        # correctness check
        assert F.allclose(out_low_mem, out_gmm_sorted, atol=1e-3, rtol=1e-3)
        assert F.allclose(Hgrad_low_mem, Hgrad_gmm_sorted, atol=1e-3, rtol=1e-3)
        assert F.allclose(Wgrad_low_mem, Wgrad_gmm_sorted, atol=1e-3, rtol=1e-3)
        assert F.allclose(out_low_mem, out_gmm_unsorted, atol=1e-3, rtol=1e-3)
        assert F.allclose(Hgrad_low_mem, Hgrad_gmm_unsorted, atol=1e-3, rtol=1e-3)
        assert F.allclose(Wgrad_low_mem, Wgrad_gmm_unsorted, atol=1e-3, rtol=1e-3)

    _test(1)
    _test(4)
    _test(16)
    _test(32)

if __name__ == '__main__':
    test_gathermm()
