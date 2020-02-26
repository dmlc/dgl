import dgl
from dgl.data import RedditDataset
import torch as th
import numpy as np
from dgl.contrib.sampling import NeighborSampler
import time

data = RedditDataset(self_loop=True)
train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
g = dgl.DGLGraph(data.graph, readonly=True)
print('Homo-graph created')

hg = dgl.graph(data.graph.edges())
print('hetero-graph created')

# try run
f1 = dgl.sampling.sample_neighbors(hg, train_nid[0:10], 2)
print('Dry run finished')

#print(train_nid[0:1000])

##################### Test 1: Use the new subgraph sampling API ####################
ne = []
t = time.time()
for i in range(100):
    seed_nodes = train_nid[i*1000:(i+1)*1000]
    f1 = dgl.sampling.sample_neighbors(hg, seed_nodes, 10, replace=False)
    u, _ = f1.edges(form='uv')
    f2 = dgl.sampling.sample_neighbors(hg, th.unique(u), 10, replace=False)
    ne.append(f1.number_of_edges() + f2.number_of_edges())
    f1, f2 = dgl.compact_graphs([f1, f2])
    #print(i, f2.number_of_nodes(), f2.number_of_edges())
dur = time.time() - t
print('Time:', dur)
print('Tput(KETPS):', np.average(ne) / dur / 1000)

##################### Test 1.5: Use the new subgraph sampling API + multi-process dataloader ####################
ne = []
t = time.time()
for i in range(100):
    seed_nodes = train_nid[i*1000:(i+1)*1000]
    f1 = dgl.sampling.sample_neighbors(hg, seed_nodes, 10, replace=False)
    u, _ = f1.edges(form='uv')
    f2 = dgl.sampling.sample_neighbors(hg, th.unique(u), 10, replace=False)
    ne.append(f1.number_of_edges() + f2.number_of_edges())
    #f1, f2 = dgl.compact_graphs([f1, f2])
    #print(i, f2.number_of_nodes(), f2.number_of_edges())
dur = time.time() - t
print('Time:', dur)
print('Tput(KETPS):', np.average(ne) / dur / 1000)

##################### Test 2: Use the old sampler data loader ####################
sampler = NeighborSampler(
        g, 1000, 10,
        neighbor_type='in',
        shuffle=False,
        num_hops=2,
        seed_nodes=train_nid,
        num_workers=4)

# try run
for nf in sampler:
    break
print('Dry run finished')

ne = []
t = time.time()
for i, nf in enumerate(sampler):
    #print(i, len(nf.block_eid(0)))
    ne.append(sum([nf.block_size(j) for j in range(2)]))
    if i == 99:
        break
dur = time.time() - t
print('Time:', dur)
print('Tput(KETPS):', np.average(ne) / dur / 1000)

##################### Test 3: PyG ####################
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler as NSPyG
import os.path as osp
path = osp.join(osp.dirname(osp.realpath(__file__)), '_tmp', 'Reddit')
dataset = Reddit(path)
data = dataset[0]
loader = NSPyG(data, size=[10, 10], num_hops=2, batch_size=1000,
                          shuffle=False, add_self_loops=False)(data.train_mask)

# dry run
for df in loader:
    break
print('Dry run finished')

ne = []
t = time.time()
for i, df in enumerate(loader):
    ne.append(df[0].edge_index.shape[1] + df[1].edge_index.shape[1])
    if i == 99:
        break
dur = time.time() - t
print('Time:', dur)
print('Tput(KETPS):', np.average(ne) / dur / 1000)
