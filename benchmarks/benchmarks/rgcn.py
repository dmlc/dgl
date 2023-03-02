import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

from . import utils


class RGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        regularizer="basis",
        num_bases=-1,
        dropout=0.0,
        self_loop=False,
        ns_mode=False,
    ):
        super(RGCN, self).__init__()

        if num_bases == -1:
            num_bases = num_rels
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.conv1 = RelGraphConv(
            h_dim, h_dim, num_rels, regularizer, num_bases, self_loop=self_loop
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer,
            num_bases,
            self_loop=self_loop,
        )
        self.dropout = nn.Dropout(dropout)
        self.ns_mode = ns_mode

    def forward(self, g, nids=None):
        if self.ns_mode:
            # forward for neighbor sampling
            x = self.emb(g[0].srcdata[dgl.NID])
            h = self.conv1(g[0], x, g[0].edata[dgl.ETYPE], g[0].edata["norm"])
            h = self.dropout(F.relu(h))
            h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], g[1].edata["norm"])
            return h
        else:
            x = self.emb.weight if nids is None else self.emb(nids)
            h = self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"])
            h = self.dropout(F.relu(h))
            h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
            return h


def load_data(data_name, get_norm=False, inv_target=False):
    dataset = utils.process_data(data_name)

    # Load hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    labels = hg.nodes[category].data.pop("labels")
    train_mask = hg.nodes[category].data.pop("train_mask")
    test_mask = hg.nodes[category].data.pop("test_mask")
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    if get_norm:
        # Calculate normalization weight for each edge,
        # 1. / d, d is the degree of the destination node
        for cetype in hg.canonical_etypes:
            hg.edges[cetype].data["norm"] = dgl.norm_by_dst(
                hg, cetype
            ).unsqueeze(1)
        edata = ["norm"]
    else:
        edata = None

    # get target category id
    category_id = hg.ntypes.index(category)

    g = dgl.to_homogeneous(hg, edata=edata)
    # Rename the fields as they can be changed by for example DataLoader
    g.ndata["ntype"] = g.ndata.pop(dgl.NTYPE)
    g.ndata["type_id"] = g.ndata.pop(dgl.NID)
    node_ids = torch.arange(g.num_nodes())

    # find out the target node ids in g
    loc = g.ndata["ntype"] == category_id
    target_idx = node_ids[loc]

    if inv_target:
        # Map global node IDs to type-specific node IDs. This is required for
        # looking up type-specific labels in a minibatch
        inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64)
        inv_target[target_idx] = torch.arange(
            0, target_idx.shape[0], dtype=inv_target.dtype
        )
        return (
            g,
            num_rels,
            num_classes,
            labels,
            train_idx,
            test_idx,
            target_idx,
            inv_target,
        )
    else:
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx
