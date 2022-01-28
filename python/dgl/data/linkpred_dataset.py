from ..convert import graph
from ..sampling.negative import _calc_redundancy
import numpy as np
import torch


def negative_sample(g, num_samples, seed):
    num_nodes = g.num_nodes()
    redundancy = _calc_redundancy(
        num_samples, g.num_edges(), num_nodes ** 2)
    sample_size = int(num_samples*(1+redundancy))
    rng = np.random.default_rng(seed)
    edges = torch.as_tensor(rng.integers(0, num_nodes, size=(2, sample_size)))
    edges = torch.unique(edges, dim=1)
    mask_self_loop = edges[0] == edges[1]
    has_edges = g.has_edges_between(edges[0], edges[1])
    mask = ~(torch.logical_or(mask_self_loop, has_edges))
    edges = edges[:, mask]
    if edges.shape[1] >= num_samples:
        edges = edges[:, :num_samples]
    return edges


class LinkPredDataset:
    def __init__(self, dataset, neg_sample_ratio: int = 3, seed=42, train_val_test_ratio=[0.8, 0.1, 0.1]):
        HAVE_OGB = False
        try:
            import ogb
            HAVE_OGB = True
        except:
            pass

        self.ds = dataset
        self._num_nodes = dataset[0].num_nodes()
        self.feat = dataset[0].ndata["feat"]
        if HAVE_OGB:
            from ogb.linkproppred import DglLinkPropPredDataset
            if isinstance(dataset, DglLinkPropPredDataset):
                self.is_ogb_dataset = True
                self.edge_split = self.ds.get_edge_split()
                self.train_graph = self.ds[0]
                self.train_graph.ndata['feat'] = self.feat
                pos_e_tensor, neg_e_tensor = self.edge_split["valid"][
                    "edge"], self.edge_split["valid"]["edge_neg"]
                pos_e = (pos_e_tensor[:, 0], pos_e_tensor[:, 1])
                neg_e = (neg_e_tensor[:, 0], neg_e_tensor[:, 1])
                self.val_edges = pos_e, neg_e
                pos_e_tensor, neg_e_tensor = self.edge_split["test"][
                    "edge"], self.edge_split["test"]["edge_neg"]
                pos_e = (pos_e_tensor[:, 0], pos_e_tensor[:, 1])
                neg_e = (neg_e_tensor[:, 0], neg_e_tensor[:, 1])
                self.test_edges = pos_e, neg_e
                return

        g = dataset[0]
        src, dst = g.edges()
        num_edges = g.num_edges()
        num_nodes = g.num_nodes()

        train_val_test_ratio = np.array(
            train_val_test_ratio) / np.sum(train_val_test_ratio)
        train_size, val_size = int(
            num_edges*train_val_test_ratio[0]), int(num_edges*train_val_test_ratio[1])
        test_size = num_edges - train_size - val_size
        idx = torch.randperm(
            g.num_edges(), generator=torch.Generator().manual_seed(seed))
        train_pos_idx = idx[:train_size]
        val_pos_idx = idx[train_size:train_size+val_size]
        test_pos_idx = idx[train_size+val_size:]
        neg_src, neg_dst = negative_sample(g, val_size+test_size, seed)
        neg_val_src, neg_val_dst = neg_src[:val_size], neg_dst[:val_size]
        neg_test_src, neg_test_dst = neg_src[val_size:], neg_dst[val_size:]
        self.val_edges = (src[val_pos_idx], dst[val_pos_idx]
                          ), (neg_val_src, neg_val_dst)
        self.train_graph = graph(
            (src[train_pos_idx], dst[train_pos_idx]), num_nodes=self.num_nodes)
        self.train_graph.ndata["feat"] = g.ndata["feat"]
        self.test_edges = (src[test_pos_idx],
                           dst[test_pos_idx]), (neg_test_src, neg_test_dst)

    @property
    def feat_size(self):
        return self.feat.shape[-1]

    @property
    def num_nodes(self):
        return self._num_nodes

    def get_val_edges(self):
        return self.val_edges

    def get_train_graph(self):
        return self.train_graph

    def get_test_edges(self):
        return self.test_edges

    # def
