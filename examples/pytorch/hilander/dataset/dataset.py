import pickle

import numpy as np
import torch
from utils import (
    build_knns,
    build_next_level,
    decode,
    density_estimation,
    fast_knns2spmat,
    knns2ordered_nbrs,
    l2norm,
    row_normalize,
    sparse_mx_to_indices_values,
)

import dgl


class LanderDataset(object):
    def __init__(
        self,
        features,
        labels,
        cluster_features=None,
        k=10,
        levels=1,
        faiss_gpu=False,
    ):
        self.k = k
        self.gs = []
        self.nbrs = []
        self.dists = []
        self.levels = levels

        # Initialize features and labels
        features = l2norm(features.astype("float32"))
        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.long)
        ids = np.arange(global_num_nodes)

        # Recursive graph construction
        for lvl in range(self.levels):
            if features.shape[0] <= self.k:
                self.levels = lvl
                break
            if faiss_gpu:
                knns = build_knns(features, self.k, "faiss_gpu")
            else:
                knns = build_knns(features, self.k, "faiss")
            dists, nbrs = knns2ordered_nbrs(knns)
            self.nbrs.append(nbrs)
            self.dists.append(dists)
            density = density_estimation(dists, nbrs, labels)

            g = self._build_graph(
                features, cluster_features, labels, density, knns
            )
            self.gs.append(g)

            if lvl >= self.levels - 1:
                break

            # Decode peak nodes
            (
                new_pred_labels,
                peaks,
                global_edges,
                global_pred_labels,
                global_peaks,
            ) = decode(
                g,
                0,
                "sim",
                True,
                ids,
                global_edges,
                global_num_nodes,
                global_peaks,
            )
            ids = ids[peaks]
            features, labels, cluster_features = build_next_level(
                features,
                labels,
                peaks,
                global_features,
                global_pred_labels,
                global_peaks,
            )

    def _build_graph(self, features, cluster_features, labels, density, knns):
        adj = fast_knns2spmat(knns, self.k)
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)

        g = dgl.graph((indices[1], indices[0]))
        g.ndata["features"] = torch.FloatTensor(features)
        g.ndata["cluster_features"] = torch.FloatTensor(cluster_features)
        g.ndata["labels"] = torch.LongTensor(labels)
        g.ndata["density"] = torch.FloatTensor(density)
        g.edata["affine"] = torch.FloatTensor(values)
        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata["global_eid"] = g.edges(form="eid")
        g.ndata["norm"] = torch.FloatTensor(adj_row_sum)
        g.apply_edges(
            lambda edges: {
                "raw_affine": edges.data["affine"] / edges.dst["norm"]
            }
        )
        g.apply_edges(
            lambda edges: {
                "labels_conn": (
                    edges.src["labels"] == edges.dst["labels"]
                ).long()
            }
        )
        g.apply_edges(
            lambda edges: {
                "mask_conn": (
                    edges.src["density"] > edges.dst["density"]
                ).bool()
            }
        )
        return g

    def __getitem__(self, index):
        assert index < len(self.gs)
        return self.gs[index]

    def __len__(self):
        return len(self.gs)
