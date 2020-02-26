import dgl
from dgl.data import RedditDataset
import torch as th
import numpy as np
from dgl.contrib.sampling import NeighborSampler
import time

data = RedditDataset(self_loop=True)
train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
num_train = len(train_nid)
g = dgl.DGLGraph(data.graph, readonly=True)
print('Homo-graph created')

hg = dgl.graph(data.graph.edges())
print('hetero-graph created')

#print(train_nid[0:1000])

L = 4
FAN = 10

##################### Test 1: Use the new subgraph sampling API ####################
ne = []
compact_dur = 0.
for i in range(150):
    if i == 20:
        print('Dry run finished')
        t = time.time()
    seed_nodes = train_nid[(i*1000)%num_train:((i+1)*1000)%num_train]
    frs = []
    for j in range(L):
        f1 = dgl.sampling.sample_neighbors(hg, seed_nodes, FAN, replace=False)
        u, _ = f1.edges(form='uv')
        seed_nodes = th.unique(u)
        frs.append(f1)
        if i >= 20:
            ne.append(f1.number_of_edges())
    tt = time.time()
    frs = dgl.compact_graphs(frs)
    compact_dur += time.time() - tt
dur = time.time() - t
print('Time:', dur)
print('v05 Tput(KETPS):', np.sum(ne) / dur / 1000)
print('v05 (w/o compact) Tput(KETPS):', np.sum(ne) / (dur - compact_dur) / 1000)

exit(1)

##################### Test 1.5: Use the new subgraph sampling API + multi-process dataloader ####################
#ne = []
#t = time.time()
#for i in range(100):
#    seed_nodes = train_nid[i*1000:(i+1)*1000]
#    f1 = dgl.sampling.sample_neighbors(hg, seed_nodes, 10, replace=False)
#    u, _ = f1.edges(form='uv')
#    f2 = dgl.sampling.sample_neighbors(hg, th.unique(u), 10, replace=False)
#    ne.append(f1.number_of_edges() + f2.number_of_edges())
#    #f1, f2 = dgl.compact_graphs([f1, f2])
#    #print(i, f2.number_of_nodes(), f2.number_of_edges())
#dur = time.time() - t
#print('Time:', dur)
#print('Tput(KETPS):', np.average(ne) / dur / 1000)

##################### Test 2: Use the old sampler data loader ####################
sampler = NeighborSampler(
        g, 1000, FAN,
        neighbor_type='in',
        shuffle=False,
        num_hops=L,
        seed_nodes=train_nid,
        num_workers=64)

ne = []
for i, nf in enumerate(sampler):
    if i == 20:
        print('Dry run finished')
        t = time.time()
    #print(i, len(nf.block_eid(0)))
    if i >= 20:
        ne.append(sum([nf.block_size(j) for j in range(L)]))
    if i == 149:
        break
print(i)
dur = time.time() - t
print('Time:', dur)
print('NF Tput(KETPS):', np.sum(ne) / dur / 1000)

exit(0)

##################### Test 3: PyG ####################
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler as NSPyG
import os.path as osp
path = osp.join(osp.dirname(osp.realpath(__file__)), '_tmp', 'Reddit')
dataset = Reddit(path)
data = dataset[0]
loader = NSPyG(data, size=FAN, num_hops=L, batch_size=1000,
                          shuffle=False, add_self_loops=False)(data.train_mask)

ne = []
for i, df in enumerate(loader):
    if i == 20:
        print('Dry run finished')
        t = time.time()
    if i >= 20:
        ne.append(sum(df[j].edge_index.shape[1] for j in range(L)))
    if i == 149:
        break
print(i)
dur = time.time() - t
print('Time:', dur)
print('Tput(KETPS):', np.sum(ne) / dur / 1000)
