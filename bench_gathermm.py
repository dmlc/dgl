from timeit import default_timer
import torch as th
import dgl
import dgl.backend as F
import dgl.function as fn
import torch.nn as nn
import time
import numpy as np

#####################################
# Timer code from benchmarks folder
#####################################
class Timer:
    def __init__(self, device):
        self.timer = default_timer
        self.device = device

    def __enter__(self):
        if self.device == 'cuda:0':
            self.start_event = th.cuda.Event(enable_timing=True)
            self.end_event = th.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == 'cuda:0':
            self.end_event.record()
            th.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(
                self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic

iters = 35
feat_scale = 1
n_edge_scale = 1
num_rel_scale = 1

in_feat = 16 * feat_scale
out_feat = 16 * feat_scale
th.cuda.set_device("cuda:0")

print("in/out feat", in_feat, out_feat)

E_per_rel = th.tensor([50, 100, 20, 284, 89, 10, 82, 9200, 10, 20, 30, 100,
    128, 20, 284, 89, 10, 82, 92, 10, 20, 30, 100, 1280, 20, 284, 89, 1000, 82,
    92, 10, 2000, 30, 100, 128, 20, 284, 89, 10, 82, 92, 10, 20, 30])
E_per_rel *= n_edge_scale
#E_per_rel = E_per_rel[:11].repeat(num_rel_scale)

num_rel = len(E_per_rel)
print('num_rel', num_rel)

H_arr = []
W_arr = []
Out_arr = []
Out_grad_arr = []

for eid in range(num_rel):
    H_arr.append(th.rand(E_per_rel[eid], in_feat).to("cuda:0"))
    W_arr.append(th.rand(in_feat, out_feat).to("cuda:0"))
    Out_arr.append(th.zeros(E_per_rel[eid], out_feat).to("cuda:0"))
    Out_grad_arr.append(th.rand(E_per_rel[eid], out_feat).to("cuda:0"))

H = th.cat([h for h in H_arr], 0)
W = th.cat([w for w in W_arr], 0)
W_3D = W.reshape(num_rel, in_feat, out_feat)
Out = th.cat([out for out in Out_arr], 0)
Out_grad = th.cat([o for o in Out_grad_arr], 0)

print('H.shape', H.shape)
print('W.shape', W.shape)
print('W_3D.shape', W_3D.shape)
print('Out.shape', Out.shape)

# **** low-mem ******

th.cuda.synchronize()
low_mem_time = []
for it in range(iters):
    with Timer('cuda:0') as t:
        out = []
        for i in range(len(E_per_rel)):
            Hi = H_arr[i]
            Wi = W_arr[i]
            out.append(th.matmul(Hi, Wi))
        out_low_mem = th.cat(out, 0)
    if it > 5:
        low_mem_time.append(t.elapsed_secs)
print("low-mem matmul:", np.average(low_mem_time)*1000, "ms")

# backward
th.cuda.synchronize()
low_mem_bw_time = []
for it in range(iters):
    with Timer('cuda:0') as t:
        H_grad = []
        W_grad = []
        for i in range(len(E_per_rel)):
            Hi = H_arr[i]
            Wi = W_arr[i] #th.ones((E_per_rel[i], out_feat), dtype=th.float32, device='cuda:0')
            Out_gradi = Out_grad_arr[i]
            H_grad.append(th.matmul(Out_gradi, Wi.transpose(0,1)))
            W_grad.append(th.matmul(Hi.transpose(0,1), Out_gradi))
        Hgrad_low_mem = th.cat(H_grad, 0)
        Wgrad_low_mem = th.cat(W_grad, 0)
    if it > 5:
        low_mem_bw_time.append(t.elapsed_secs)
print("low-mem backward matmul:", np.average(low_mem_bw_time)*1000, "ms")

# **** high-mem ******

th.cuda.synchronize()
etype_arr = []
for eid in range(num_rel):
    etype_arr.append(th.full((E_per_rel[eid],), eid).to("cuda:0"))
etypes = th.cat([etype for etype in etype_arr], 0)

high_mem_time = []
for i in range(iters):
    with Timer('cuda:0') as t:
        W_high_mem = W_3D.index_select(0, etypes)
        out_high_mem = th.bmm(H.unsqueeze(1), W_high_mem).squeeze(1)
    if i > 5:
        high_mem_time.append(t.elapsed_secs)
# print("high-mem matmul:", np.average(high_mem_time)*1000, "ms")


# **** Sorted Gather mm ******
th.cuda.synchronize()
_gather_mm_time = []
for i in range(iters):
    with Timer('cuda:0') as t:
        out_gmm_sorted = th.zeros(Out.shape, dtype=th.float32, device='cuda:0')
        dgl.sparse._gather_mm(H, W, out_gmm_sorted, E_per_rel, etypes, sortedE=True)
    if i > 5:
        _gather_mm_time.append(t.elapsed_secs)
print("gather_mm Sorted:", np.average(_gather_mm_time)*1000, "ms")

# backward
th.cuda.synchronize()
_gather_mm_bw_time = []
for i in range(iters):
    with Timer('cuda:0') as t:
        # Compute H_grad
        W_3D_trans = th.transpose(W_3D, 1, 2)
        W_2D_trans = W_3D_trans.reshape(num_rel * out_feat, in_feat)
        H_grad = th.zeros(H.shape, dtype=th.float32, device='cuda:0')
        dgl.sparse._gather_mm_scatter(Out_grad, W_2D_trans, H_grad, E_per_rel, etypes, sortedE=True)
        # Compute W_grad
        W_grad = th.zeros(W.shape, dtype=th.float32, device='cuda:0')
        dgl.sparse._gather_mm_scatter(H, Out_grad, W_grad, E_per_rel, etypes, sortedE=True, h_trans=True)
    if i > 5:
        _gather_mm_bw_time.append(t.elapsed_secs)
print("gather_mm Sorted backward:", np.average(_gather_mm_bw_time)*1000, "ms")

# **** Unsorted Gather mm ******
th.cuda.synchronize()
_gather_mm_time_un = []
for i in range(iters):
    with Timer('cuda:0') as t:
        out_gmm_unsorted = th.zeros(Out.shape, dtype=th.float32, device='cuda:0')
        dgl.sparse._gather_mm(H, W, out_gmm_unsorted, E_per_rel, etypes, sortedE=False)
        # print("gather-mm-mem Unsorted", out_gmm_unsorted)
    if i > 5:
        _gather_mm_time_un.append(t.elapsed_secs)
# print("gather_mm Unsorted:", np.average(_gather_mm_time_un)*1000, "ms")


assert th.allclose(out_low_mem, out_high_mem, atol=1e-3, rtol=1e-3)
assert th.allclose(out_low_mem, out_gmm_sorted, atol=1e-3, rtol=1e-3)
assert th.allclose(out_high_mem, out_gmm_unsorted, atol=1e-3, rtol=1e-3)
assert th.allclose(Hgrad_low_mem, H_grad, atol=1e-3, rtol=1e-3)
assert th.allclose(Wgrad_low_mem, W_grad, atol=1e-3, rtol=1e-3)
