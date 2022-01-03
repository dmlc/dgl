from ogb.nodeproppred import DglNodePropPredDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h

dataset = DglNodePropPredDataset('ogbn-products')
graph, labels = dataset[0]
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

sampler = dglnew.dataloading.MultiLayerNeighborSampler([5, 5, 5])
dataloader = dglnew.dataloading.NodeDataLoader(
        graph,
        train_idx,
        sampler,
        device='cuda',
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        num_workers=4)
sampler.add_input('feat', g.ndata['feat'])
sampler.add_output('label', g.ndata['label'])

model = SAGE(graph.ndata['feat'].shape[1], 128, dataset.num_classes)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

for input_nodes, seeds, blocks in dataloader:
    x = blocks[0].srcdata['feat']
    y = blocks[-1].dstdata['label']
    y_hat = model(blocks, x)
    loss = F.cross_entropy(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
