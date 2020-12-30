import time
import numpy as np
import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils

class RGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 n_hidden,
                 num_classes,
                 num_rels,
                 num_bases,
                 num_hidden_layers,
                 dropout,
                 lowmem):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(num_nodes, n_hidden, num_rels, "basis",
                                        num_bases, activation=F.relu, dropout=dropout,
                                        low_mem=lowmem))
        # h2h
        for i in range(num_hidden_layers):
            self.layers.append(RelGraphConv(n_hidden, n_hidden, num_rels, "basis",
                                            num_bases, activation=F.relu, dropout=dropout,
                                            low_mem=lowmem))
        # o2h
        self.layers.append(RelGraphConv(n_hidden, num_classes, num_rels, "basis",
                                        num_bases, activation=None, low_mem=lowmem))

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

def sort_graph(g, etypes, num_rels):
    # Sort the graph based on the etypes
    sorted_etypes, index = torch.sort(etypes)
    g = dgl.edge_subgraph(g, index, preserve_nodes=True)
    # Create a new etypes to be an integer list of number of edges.
    pos = _searchsorted(sorted_etypes, torch.arange(num_rels, device=g.device), num_rels)
    num = torch.tensor([len(etypes)], device=g.device)
    etypes = (torch.cat([pos[1:], num]) - pos).tolist()
    return g, etypes

def _searchsorted(sorted_sequence, values, num_rels):
    # searchsorted is introduced to PyTorch in 1.6.0
    if getattr(torch, 'searchsorted', None):
        return torch.searchsorted(sorted_sequence, values)
    else:
        device = values.device
        return torch.from_numpy(np.searchsorted(sorted_sequence.cpu().numpy(),
                                                values.cpu().numpy())).to(device)

@utils.benchmark('time', 3600)
@utils.parametrize('data', ['aifb', 'am'])
@utils.parametrize('lowmem', [True, False])
@utils.parametrize('sortgraph', [True, False])
def track_time(data, lowmem, sortgraph):
    # args
    if data == 'aifb':
        num_bases = -1
        l2norm = 0.
    elif data == 'am':
        num_bases = 40
        l2norm = 5e-4
    else:
        raise ValueError()

    data = utils.process_data(data)
    device = utils.get_bench_device()
    num_epochs = 30

    g = data[0]

    num_rels = len(g.canonical_etypes)
    category = data.predict_category
    num_classes = data.num_classes
    train_mask = g.nodes[category].data.pop('train_mask').bool().to(device)
    test_mask = g.nodes[category].data.pop('test_mask').bool().to(device)
    labels = g.nodes[category].data.pop('labels').to(device)
    
    # calculate norm for each edge type and store in edge
    for canonical_etype in g.canonical_etypes:
        u, v, eid = g.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = 1. / degrees.float()
        norm = norm.unsqueeze(1)
        g.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    g = dgl.to_homogeneous(g, edata=['norm']).to(device)
    num_nodes = g.number_of_nodes()
    edge_type = g.edata[dgl.ETYPE].long()

    if sortgraph:
        # sort graph 
        g, edge_type = sort_graph(g, edge_type, num_rels)

    edge_norm = g.edata['norm']

    # find out the target node ids in g
    target_idx = torch.where(g.ndata[dgl.NTYPE] == category_id)[0]
    train_idx = target_idx[train_mask]
    test_idx = target_idx[test_mask]
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes, device=device)

    # create model
    model = RGCN(num_nodes, 
                 16,
                 num_classes,
                 num_rels,
                 num_bases,
                 0,
                 0,
                 lowmem).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=l2norm)

    model.train()
    t0 = time.time()
    for epoch in range(num_epochs):
        logits = model(g, feats, edge_type, edge_norm)
        loss = F.cross_entropy(logits[train_idx], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t1 = time.time()

    return (t1 - t0) / num_epochs
