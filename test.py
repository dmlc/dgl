import dgl
from dgl.data import RedditDataset
import torch as th
import numpy as np
from dgl.contrib.sampling import NeighborSampler
import time

data = RedditDataset(self_loop=True)
train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
num_train = len(train_nid)
print('#Train nodes:', num_train)
g = dgl.DGLGraph(data.graph, readonly=True)
print('Homo-graph created')

hg = dgl.graph(data.graph.edges())
print('hetero-graph created')

#print(train_nid[0:1000])

L = 2
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

##################### Test 1.5: Use the new subgraph sampling API + multi-process dataloader ####################
import multiprocessing as mp

class MyNeighborSampler:
    def __init__(self, g, seed_nodes, fanout, hops, replace, batch_size, num_workers=1):
        self.g = g
        self.seed_nodes = seed_nodes
        if not isinstance(fanout, list):
            fanout = [fanout] * hops
        self.fanout = fanout
        self.hops = hops
        self.replace = replace
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = mp.Queue()
        self.workers = []

    def __iter__(self):
        if len(self.workers) == 0:
            self.start()
        return iter(self.queue.get, None)

    def __del__(self):
        for worker in self.workers: 
            worker.join()

    def start(self):
        # split seeds
        splits = self.seed_nodes.split(self.batch_size)
        per_worker = (len(splits) + self.num_workers) // self.num_workers
        for i in range(self.num_workers):
            workload = splits[i * per_worker : (i+1) * per_worker]
            worker = mp.Process(target=self.worker_loop, args=(i, workload))
            worker.start()
            self.workers.append(worker)

    def wait(self):
        for worker in self.workers: 
            worker.join()

    def worker_loop(self, pid, workload):
        print('Process #%d launched, #works=%d' % (pid, len(workload)))
        for seeds in workload:
            frs = []
            for i in range(self.hops):
                fr = dgl.sampling.sample_neighbors(self.g, seeds, self.fanout[i], replace=self.replace)
                u, _ = fr.edges(form='uv')
                seeds = th.unique(u)
                frs.append(fr)
            frs = dgl.compact_graphs(frs)
            #self.queue.put(frs)
            self.queue.put(sum([frs[j].number_of_edges() for j in range(self.hops)]))
        print('Process #%d finished' % pid)

sampler_x = MyNeighborSampler(
        hg, train_nid, FAN, L, False, 1000,
        num_workers=4)

ne = []
for i, frs in enumerate(sampler_x):
    if i == 20:
        print('Dry run finished')
        t = time.time()
    #print(i, len(nf.block_eid(0)))
    if i >= 20:
        #ne.append(sum([frs[j].number_of_edges() for j in range(L)]))
        ne.append(frs)
    if i == 149:
        break
print(i)
dur = time.time() - t
print('Time:', dur)
print('v05 MP Tput(KETPS):', np.sum(ne) / dur / 1000)

##################### Test 2: Use the old sampler data loader ####################
sampler = NeighborSampler(
        g, 1000, FAN,
        neighbor_type='in',
        shuffle=False,
        num_hops=L,
        seed_nodes=train_nid,
        num_workers=4)

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
