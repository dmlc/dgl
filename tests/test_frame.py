import torch as F
from dgl.array import DGLDenseArray
from dgl.dense_aggregate import COUNT, SUM
from dgl.frame import DGLFrame

N_NODES = 100
N_EDGES = 1000
N_FEATS = 10

idx = F.arange(N_NODES).long()
idx_array = DGLDenseArray(idx)
dat = F.randn(N_NODES, N_FEATS)
dat_array = DGLDenseArray(dat)
node_frame = DGLFrame({'idx' : idx_array, 'dat' : dat_array})
mask = DGLDenseArray(F.max(node_frame['dat'].data, 1)[0] > 1)
node_frame = node_frame[mask]
assert F.prod(F.max(node_frame['dat'].data, 1)[0] > 1)

src = F.randint(N_NODES, [N_EDGES]).long()
dst = F.randint(N_NODES, [N_EDGES]).long()
dat = F.randn(N_EDGES, N_FEATS)
src_array = DGLDenseArray(src)
dst_array = DGLDenseArray(dst)
dat_array = DGLDenseArray(dat)
edge_frame = DGLFrame({'src' : src_array, 'dst' : dst_array, 'dat' : dat_array})

# filterby_frame = edge_frame.filter_by(s

count_frame = edge_frame.groupby('dst', {'count' : COUNT()})
for x, y in zip(count_frame['dst'].values(), count_frame['count'].values()):
    assert F.sum(dst == x) == y
    
sum_frame = edge_frame.groupby('dst', {'sum' : SUM('dat')})
for x, y in zip(count_frame['dst'].values(), count_frame['sum'].values()):
    assert F.sum(dat[dst == x]) == y
