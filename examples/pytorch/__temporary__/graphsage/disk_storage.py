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

import dglnew

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
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

graph.create_formats_()

# This is an example of using feature storage other than tensors
feat_np = graph.ndata['feat'].numpy()
feat = np.memmap('feat.npy', mode='w+', shape=feat_np.shape, dtype='float32')
print(feat.shape)
feat[:] = feat_np
graph = dglnew.graph.OtherFeatureGraphStorage(graph,
        ndata={'feat': dglnew.storages.NumpyStorage(feat), 'label': labels})

sampler = dglnew.dataloading.NeighborSampler([5, 5, 5], output_device='cpu')
sampler.add_input('feat')
sampler.add_output('label')
dataloader = dglnew.dataloading.NodeDataLoader(
        graph,
        train_idx,
        sampler,
        device='cuda',
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
        use_asyncio=True,
        use_prefetch_thread=True)       # TBD: could probably remove this argument

model = SAGE(feat.shape[1], 256, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

durations = []
for _ in range(10):
    t0 = time.time()
    for it, blocks in enumerate(dataloader):
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
