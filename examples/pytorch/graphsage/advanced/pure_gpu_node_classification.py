import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset

import dgl
import dgl.nn as dglnn


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        feat = g.ndata["feat"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device,
                pin_memory=True,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # use an explicitly contuous slice
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # be design, our output nodes are contiguous so we can take
                # advantage of that here
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


dataset = DglNodePropPredDataset("ogbn-products")
graph, labels = dataset[0]
graph.ndata["label"] = labels.squeeze()
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = (
    split_idx["train"],
    split_idx["valid"],
    split_idx["test"],
)

device = "cuda"
train_idx = train_idx.to(device)
valid_idx = valid_idx.to(device)
test_idx = test_idx.to(device)

graph = graph.to(device)

model = SAGE(graph.ndata["feat"].shape[1], 256, dataset.num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

sampler = dgl.dataloading.NeighborSampler(
    [15, 10, 5], prefetch_node_feats=["feat"], prefetch_labels=["label"]
)
train_dataloader = dgl.dataloading.DataLoader(
    graph,
    train_idx,
    sampler,
    device=device,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=False,
)
valid_dataloader = dgl.dataloading.DataLoader(
    graph,
    valid_idx,
    sampler,
    device=device,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=False,
)

durations = []
for _ in range(10):
    model.train()
    t0 = time.time()
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            acc = MF.accuracy(torch.argmax(y_hat, dim=1), y)
            mem = torch.cuda.max_memory_allocated() / 1000000
            print("Loss", loss.item(), "Acc", acc.item(), "GPU Mem", mem, "MB")
    tt = time.time()
    print(tt - t0)
    durations.append(tt - t0)

    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(torch.argmax(model(blocks, x), dim=1))
    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
    print("Validation acc:", acc.item())

print(np.mean(durations[4:]), np.std(durations[4:]))

# Test accuracy and offline inference of all nodes
model.eval()
with torch.no_grad():
    pred = model.inference(graph, device, 4096, 0, "cpu")
    pred = pred[test_idx].to(device)
    label = graph.ndata["label"][test_idx]
    acc = MF.accuracy(torch.argmax(pred, dim=1), label)
    print("Test acc:", acc.item())
