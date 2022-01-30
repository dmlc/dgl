import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
# OGB must follow DGL if both DGL and PyG are installed. Otherwise DataLoader will hang.
# (This is a long-standing issue)
from ogb.nodeproppred import DglNodePropPredDataset

USE_WRAPPER = True

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        with pair_graph.local_scope(), neg_pair_graph.local_scope():
            pair_graph.ndata['h'] = neg_pair_graph.ndata['h'] = h
            pair_graph.apply_edges(dgl.function.u_dot_v('h', 'h', 's'))
            neg_pair_graph.apply_edges(dgl.function.u_dot_v('h', 'h', 's'))
            return pair_graph.edata['s'], neg_pair_graph.edata['s']

dataset = DglNodePropPredDataset('ogbn-products')
graph, labels = dataset[0]
graph.ndata['label'] = labels
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

model = SAGE(graph.ndata['feat'].shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

num_edges = graph.num_edges()
train_eids = torch.arange(num_edges)
if USE_WRAPPER:
    import dglnew
    graph.create_formats_()
    graph = dglnew.graph.wrapper.DGLGraphStorage(graph)

sampler = dgl.dataloading.NeighborSampler(
        [5, 5, 5], output_device='cpu', prefetch_node_feats=['feat'],
        prefetch_labels=['label'])
dataloader = dgl.dataloading.EdgeDataLoader(
        graph,
        train_eids,
        sampler,
        device='cuda',
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        use_prefetch_thread=True,       # TBD: could probably remove this argument
        exclude='reverse_id',
        reverse_eids=torch.arange(num_edges) ^ 1,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5))

durations = []
for _ in range(10):
    t0 = time.time()
    for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata['feat']
        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        score = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(score, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            acc = MF.auroc(score, labels.long())
            mem = torch.cuda.max_memory_allocated() / 1000000
            print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
            tt = time.time()
            print(tt - t0)
            t0 = time.time()
    durations.append(tt - t0)
print(np.mean(durations[4:]), np.std(durations[4:]))
