import sys
import random
import numpy as np
from scipy.sparse.csgraph import shortest_path
import torch
from dgl.sampling import global_uniform_negative_sampling


def k_hop_subgraph(src, dst, num_hops, g, sample_ratio=1., directed=False):
    # Extract the k-hop enclosing subgraph around link (src, dst) from g
    nodes = [src, dst]
    visited = set([src, dst])
    fringe = set([src, dst])
    for _ in range(num_hops):
        if not directed:
            _, fringe = g.out_edges(list(fringe))
            fringe = fringe.tolist()
        else:
            _, out_neighbors = g.out_edges(list(fringe))
            in_neighbors, _ = g.in_edges(list(fringe))
            fringe = in_neighbors.tolist() + out_neighbors.tolist()
        fringe = set(fringe) - visited
        visited = visited.union(fringe)
        
        if sample_ratio < 1.:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if len(fringe) == 0:
            break

        nodes = nodes + list(fringe)

    subg = g.subgraph(nodes, store_ids=True)

    return subg


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    # shortest path may include inf values
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def get_pos_neg_edges(split, split_edge, g, percent=100):
    assert 'edge' in split_edge['train']

    pos_edge = split_edge[split]['edge']
    if split == 'train':
        neg_edge = torch.stack(global_uniform_negative_sampling(
            g, num_samples=pos_edge.size(0),
            exclude_self_loops=True
        ), dim=1)
    else:
        neg_edge = split_edge[split]['edge_neg']

    # sampling according to the percent param
    np.random.seed(123)
    # pos sampling
    num_pos = pos_edge.size(0)
    perm = np.random.permutation(num_pos)
    perm = perm[:int(percent / 100 * num_pos)]
    pos_edge = pos_edge[perm]
    # neg sampling
    if neg_edge.dim() > 2: # [Np, Nn, 2]
        neg_edge = neg_edge[perm].view(-1, 2)
    else:
        np.random.seed(123)
        num_neg = neg_edge.size(0)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[perm]

    return pos_edge, neg_edge # ([2, Np], [2, Nn]) -> ([Np, 2], [Nn, 2])


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # result is in the format of (val_score, test_score)
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)
