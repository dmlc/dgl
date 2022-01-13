import torch as th
import dgl
import dgl.backend as F
import dgl.function as fn
import torch.nn as nn
import time
import numpy as np

iters = 15
in_feat = 16
out_feat = 4

th.cuda.set_device("cuda:0")

E_etype = th.tensor([50, 100, 20, 284, 89, 10, 82, 9200, 10, 20, 30, 100, 128, 20, 284, 89, 10, 82, 92, 10, 20, 30,
                  100, 1280, 20, 284, 89, 1000, 82, 92, 10, 2000, 30, 100, 128, 20, 284, 89, 10, 82, 92, 10, 20, 30])

num_rel = len(E_etype)

H_arr = []
W_arr = []
Out_arr = []

for eid in range(num_rel):
    H_arr.append(th.rand(E_etype[eid], in_feat).to("cuda:0"))
    W_arr.append(th.rand(in_feat, out_feat).to("cuda:0"))
    Out_arr.append(th.zeros(E_etype[eid], out_feat).to("cuda:0"))

H = th.cat([h for h in H_arr], 0)
W = th.cat([w for w in W_arr], 0)
Out = th.cat([out for out in Out_arr], 0)

# **** low-mem ******

th.cuda.synchronize()
low_mem_time = []
for it in range(iters):
        tic = time.time()
        for i in range(len(E_etype)):
                Hi = H_arr[i]
                Wi = W_arr[i]
                Outi = th.matmul(Hi, Wi)
        if it > 5:
               low_mem_time.append(time.time() - tic)
th.cuda.synchronize()
print("low-mem matmul:", np.average(low_mem_time)*1000, "ms")

# **** high-mem ******

th.cuda.synchronize()
etype_arr = []
for eid in range(num_rel):
        etype_arr.append(th.full((E_etype[eid],), eid).to("cuda:0"))
etypes = th.cat([etype for etype in etype_arr], 0)
W_3D = W.reshape(num_rel, in_feat, out_feat)

high_mem_time = []
for i in range(iters):
        tic = time.time()
        W_high_mem = W_3D.index_select(0, etypes)
        out_high_mem = th.bmm(H.unsqueeze(1), W_high_mem).squeeze(1)
        if i > 5:
               high_mem_time.append(time.time() - tic)
th.cuda.synchronize()
print("high-mem matmul:", np.average(high_mem_time)*1000, "ms")

# g.gather_mm(edge_udf, weight)
th.cuda.synchronize()
_gather_mm_time = []
for i in range(iters):
        tic = time.time()
        dgl.sparse._gather_mm(E_etype, H, W, Out)
        if i > 5:
               _gather_mm_time.append(time.time() - tic)
th.cuda.synchronize()
print("gather matmul:", np.average(_gather_mm_time)*1000, "ms")


