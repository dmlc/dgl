import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.utils import recursive_apply
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm

USE_WRAPPER = True

class HeteroGAT(nn.Module):
    def __init__(self, etypes, in_feats, n_hidden, n_classes, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(in_feats, n_hidden // n_heads, n_heads)
            for etype in etypes}))
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(n_hidden, n_hidden // n_heads, n_heads)
            for etype in etypes}))
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(n_hidden, n_hidden // n_heads, n_heads)
            for etype in etypes}))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(n_hidden, n_classes)   # Should be HeteroLinear

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # One thing is that h might return tensors with zero rows if the number of dst nodes
            # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
            h = recursive_apply(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = recursive_apply(h, F.relu)
                h = recursive_apply(h, self.dropout)
        return self.linear(h['paper'])

dataset = DglNodePropPredDataset('ogbn-mag')

graph, labels = dataset[0]
graph.ndata['label'] = labels
# Preprocess: add reverse edges in "cites" relation, and add reverse edge types for the
# rest.
graph = dgl.AddReverse()(graph)
# Preprocess: precompute the author, topic, and institution features
graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='rev_writes')
graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='has_topic')
graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='affiliated_with')
graph.edges['cites'].data['weight'] = torch.ones(graph.num_edges('cites'))  # dummy edge weights

model = HeteroGAT(graph.etypes, graph.ndata['feat']['paper'].shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

if USE_WRAPPER:
    import dglnew
    graph.create_formats_()
    graph = dglnew.graph.wrapper.DGLGraphStorage(graph)

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

sampler = dgl.dataloading.NeighborSampler(
        [5, 5, 5], output_device='cpu',
        prefetch_node_feats={k: ['feat'] for k in graph.ntypes},
        prefetch_labels={'paper': ['label']},
        prefetch_edge_feats={'cites': ['weight']})
dataloader = dgl.dataloading.NodeDataLoader(
        graph,
        train_idx,
        sampler,
        device='cuda',
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        use_prefetch_thread=True)       # TBD: could probably remove this argument

durations = []
for _ in range(10):
    t0 = time.time()
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']['paper'][:, 0]
        assert y.min() >= 0 and y.max() < dataset.num_classes
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            acc = MF.accuracy(y_hat, y)
            mem = torch.cuda.max_memory_allocated() / 1000000
            print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
    tt = time.time()
    print(tt - t0)
    durations.append(tt - t0)
print(np.mean(durations[4:]), np.std(durations[4:]))
