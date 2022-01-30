import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
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

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

dataset = DglNodePropPredDataset('ogbn-products')
graph, labels = dataset[0]
graph.ndata['label'] = labels
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

model = SAGE(graph.ndata['feat'].shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

if USE_WRAPPER:
    import dglnew
    graph.create_formats_()
    graph = dglnew.graph.wrapper.DGLGraphStorage(graph)

sampler = dgl.dataloading.NeighborSampler(
        [5, 5, 5], output_device='cpu', prefetch_node_feats=['feat'],
        prefetch_labels=['label'])
dataloader = dgl.dataloading.NodeDataLoader(
        graph,
        train_idx,
        sampler,
        device='cuda',
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=16,
        persistent_workers=True,
        use_prefetch_thread=True)       # TBD: could probably remove this argument

durations = []
for _ in range(10):
    t0 = time.time()
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label'][:, 0]
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
