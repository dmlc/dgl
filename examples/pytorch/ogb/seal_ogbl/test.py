from multiprocessing.context import get_spawning_popen
import random
import numpy as np
import torch
import dgl
import time
from dgl.dataloading import Sampler, DataLoader
from torch.utils.data import Dataset, DataLoader as PTDataLoader
from ogb.linkproppred import DglLinkPropPredDataset
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm


class SealSampler(Sampler):
    def __init__(self, num_hops=1, sample_ratio=1., directed=False,
        prefetch_node_feats=None, prefetch_edge_feats=None):
        super().__init__()
        self.num_hops = num_hops
        self.sample_ratio = sample_ratio
        self.directed = directed
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats

    def _double_radius_node_labeling(self, adj):
        N = adj.shape[0]
        adj_wo_src = adj[range(1, N), :][:, range(1, N)]
        idx = list(range(1)) + list(range(2, N))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=0)
        dist2src = np.insert(dist2src, 1, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=0)
        dist2dst = np.insert(dist2dst, 0, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[0: 2] = 1.
        # shortest path may include inf values
        z[torch.isnan(z)] = 0.

        return z.to(torch.long)

    def sample(self, g, seed_edges):
        graphs = []
        indices = seed_edges[:, 0]
        # construct k-hop enclosing graph for each link
        for eid in seed_edges:
            _, src, dst = map(int, eid)
            # construct the enclosing graph
            visited, nodes, fringe = [np.unique([src, dst]) for _ in range(3)]
            for _ in range(self.num_hops):
                if not self.directed:
                    _, fringe = g.out_edges(fringe)
                else:
                    _, out_neighbors = g.out_edges(fringe)
                    in_neighbors, _ = g.in_edges(fringe)
                    fringe = np.union1d(in_neighbors, out_neighbors)
                fringe = np.setdiff1d(fringe, visited)
                visited = np.union1d(visited, fringe)
                if self.sample_ratio < 1.:
                    fringe = np.random.choice(fringe, 
                        int(self.sample_ratio * len(fringe)), replace=False)
                if len(fringe) == 0:
                    break
                nodes = np.union1d(nodes, fringe)
            subg = g.subgraph(nodes, store_ids=True)

            # remove edges to predict
            edges_to_remove = [
                subg.edge_ids(s, t) for s, t in [(0, 1), (1, 0)] if subg.has_edges_between(s, t)]
            subg.remove_edges(edges_to_remove)
            # add double radius node labeling
            subg.ndata['z'] = self._double_radius_node_labeling(subg.adj(scipy_fmt='csr'))
            subg_aug = subg.add_self_loop()
            if 'w' in subg.edata:
                subg_aug.edata['w'][subg.num_edges():] = torch.ones(
                    subg_aug.num_edges() - subg.num_edges())
            graphs.append(subg_aug)

        graphs = dgl.batch(graphs)
        dgl.set_src_lazy_features(graphs, self.prefetch_node_feats)
        dgl.set_edge_lazy_features(graphs, self.prefetch_edge_feats)

        return indices, graphs


class SealDataset(Dataset):
    def __init__(self, g, links, labels, num_hops=1, p=0.6, directed=False) -> None:
        super().__init__()
        self.g = g
        self.links = links
        self.labels = labels
        self.num_hops = num_hops
        self.sample_ratio = p
        self.directed = directed

    def _double_radius_node_labeling(self, adj):
        N = adj.shape[0]
        adj_wo_src = adj[range(1, N), :][:, range(1, N)]
        idx = list(range(1)) + list(range(2, N))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=0)
        dist2src = np.insert(dist2src, 1, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=0)
        dist2dst = np.insert(dist2dst, 0, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[0: 2] = 1.
        # shortest path may include inf values
        z[torch.isnan(z)] = 0.

        return z.to(torch.long)

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        src, dst = self.links[idx]
        g = self.g
        visited, nodes, fringe = [np.unique([src, dst]) for _ in range(3)]
        for _ in range(self.num_hops):
            if not self.directed:
                _, fringe = g.out_edges(fringe)
            else:
                _, out_neighbors = g.out_edges(fringe)
                in_neighbors, _ = g.in_edges(fringe)
                fringe = np.union1d(in_neighbors, out_neighbors)
            fringe = np.setdiff1d(fringe, visited)
            visited = np.union1d(visited, fringe)
            if self.sample_ratio < 1.:
                fringe = np.random.choice(fringe, 
                    int(self.sample_ratio * len(fringe)), replace=False)
            if len(fringe) == 0:
                break
            nodes = np.union1d(nodes, fringe)
        subg = g.subgraph(nodes, store_ids=True)

        # remove edges to predict
        edges_to_remove = [
            subg.edge_ids(s, t) for s, t in [(0, 1), (1, 0)] if subg.has_edges_between(s, t)]
        subg.remove_edges(edges_to_remove)
        # add double radius node labeling
        subg.ndata['z'] = self._double_radius_node_labeling(subg.adj(scipy_fmt='csr'))
        subg_aug = subg.add_self_loop()
        if 'w' in subg.edata:
            subg_aug.edata['w'][subg.num_edges():] = torch.ones(
                subg_aug.num_edges() - subg.num_edges())

        return subg_aug, self.labels[idx]


def collate_fn(batch):
    gs, ys = zip(*batch)
    gs = dgl.batch(gs)
    ys = torch.tensor(ys)

    return gs, ys

dataset = DglLinkPropPredDataset('ogbl-ppa')
split_edge = dataset.get_edge_split()
pos_edges, neg_edges = split_edge['test']['edge'], split_edge['test']['edge_neg']
edges = torch.cat([pos_edges, neg_edges], dim=0) # [Np + Nn, 2]
labels = torch.tensor([1] * len(pos_edges) + [0] * len(neg_edges)) # [Np + Nn]
indices = torch.arange(len(labels)).unsqueeze(1) # [Np + Nn, 1]

# DGL Data Loading
eids = torch.cat([indices, edges], dim=1) # [Np + Nn, 3]
graph = dataset[0]
sampler = SealSampler(num_hops=1, sample_ratio=0.6, directed=False, prefetch_node_feats=['feat'])
data_loader = DataLoader(graph, eids, sampler, batch_size=32, num_workers=16, device=0)
N = 5000
start = time.time()
for _, (indices, graphs) in tqdm(zip(range(N), data_loader)):
    pass
end = time.time()
print(f'DGL Loading {N} examples with the speed {N/(end-start):.2f} examples/sec')

# PyTorch Data Loading
start = time.time()
data = SealDataset(graph, edges, labels, num_hops=1, p=0.6, directed=False)
data_loader = PTDataLoader(data, batch_size=32, collate_fn=collate_fn, num_workers=16)
for _, (gs, ys) in tqdm(zip(range(N), data_loader)):
    gs, ys = gs.to(0), ys.to(0)
    x = gs.ndata['feat'] if 'feat' in gs.ndata else None
    z = gs.ndata['z']
    w = gs.edata['weight'] if 'weight' in gs.edata else None
end = time.time()
print(f'PyTorch Loading {N} examples with the speed {N/(end-start):.2f} examples/sec')
