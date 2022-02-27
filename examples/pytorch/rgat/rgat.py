import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl import apply_each
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm

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
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
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

model = HeteroGAT(graph.etypes, graph.ndata['feat']['paper'].shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
train_idx = apply_each(train_idx, lambda x: x.to('cuda'))
valid_idx = apply_each(valid_idx, lambda x: x.to('cuda'))
test_idx = apply_each(test_idx, lambda x: x.to('cuda'))

train_sampler = dgl.dataloading.NeighborSampler(
        [5, 5, 5],
        prefetch_node_feats={k: ['feat'] for k in graph.ntypes},
        prefetch_labels={'paper': ['label']})
valid_sampler = dgl.dataloading.NeighborSampler(
        [10, 10, 10],   # Slightly more
        prefetch_node_feats={k: ['feat'] for k in graph.ntypes},
        prefetch_labels={'paper': ['label']})
train_dataloader = dgl.dataloading.DataLoader(
        graph, train_idx, train_sampler,
        device='cuda', batch_size=1000, shuffle=True,
        drop_last=False, num_workers=0, use_uva=True)
valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_idx, valid_sampler,
        device='cuda', batch_size=1000, shuffle=False,
        drop_last=False, num_workers=0, use_uva=True)
test_dataloader = dgl.dataloading.DataLoader(
        graph, test_idx, valid_sampler,
        device='cuda', batch_size=1000, shuffle=False,
        drop_last=False, num_workers=0, use_uva=True)

def evaluate(model, dataloader):
    preds = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']['paper'][:, 0]
            y_hat = model(blocks, x)
            preds.append(y_hat.cpu())
            labels.append(y.cpu())
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        acc = MF.accuracy(preds, labels)
        return acc

durations = []
for _ in range(10):
    model.train()
    t0 = time.time()
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']['paper'][:, 0]
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

    model.eval()
    valid_acc = evaluate(model, valid_dataloader)
    test_acc = evaluate(model, test_dataloader)
    print('Validation acc:', valid_acc, 'Test acc:', test_acc)
print(np.mean(durations[4:]), np.std(durations[4:]))
