import argparse

import dgl

import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from scipy.sparse.csgraph import shortest_path


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="SEAL")
    parser.add_argument("--dataset", type=str, default="ogbl-collab")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--model", type=str, default="dgcnn")
    parser.add_argument("--gcn_type", type=str, default="gcn")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_units", type=int, default=32)
    parser.add_argument("--sort_k", type=int, default=30)
    parser.add_argument("--pooling", type=str, default="sum")
    parser.add_argument("--dropout", type=str, default=0.5)
    parser.add_argument("--hits_k", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--neg_samples", type=int, default=1)
    parser.add_argument("--subsample_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--save_dir", type=str, default="./processed")
    args = parser.parse_args()

    return args


def load_ogb_dataset(dataset):
    """
    Load OGB dataset
    Args:
        dataset(str): name of dataset (ogbl-collab, ogbl-ddi, ogbl-citation)

    Returns:
        graph(DGLGraph): graph
        split_edge(dict): split edge

    """
    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    return graph, split_edge


def drnl_node_labeling(subgraph, src, dst):
    """
    Double Radius Node labeling
    d = r(i,u)+r(i,v)
    label = 1+ min(r(i,u),r(i,v))+ (d//2)*(d//2+d%2-1)
    Isolated nodes in subgraph will be set as zero.
    Extreme large graph may cause memory error.

    Args:
        subgraph(DGLGraph): The graph
        src(int): node id of one of src node in new subgraph
        dst(int): node id of one of dst node in new subgraph
    Returns:
        z(Tensor): node labeling tensor
    """
    adj = subgraph.adj_external().to_dense().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(
        adj_wo_dst, directed=False, unweighted=True, indices=src
    )
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)


def evaluate_hits(name, pos_pred, neg_pred, K):
    """
    Compute hits
    Args:
        name(str): name of dataset
        pos_pred(Tensor): predict value of positive edges
        neg_pred(Tensor): predict value of negative edges
        K(int): num of hits

    Returns:
        hits(float): score of hits


    """
    evaluator = Evaluator(name)
    evaluator.K = K
    hits = evaluator.eval(
        {
            "y_pred_pos": pos_pred,
            "y_pred_neg": neg_pred,
        }
    )[f"hits@{K}"]

    return hits
