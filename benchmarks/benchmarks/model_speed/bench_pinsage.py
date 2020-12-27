import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl

from .. import utils

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

@utils.benchmark('time', 3600)
@utils.parametrize('data', ['nowplaying_rs'])
def track_time(data):
    data = utils.process_data(data)
    device = utils.get_bench_device()

    user_ntype = data.user_ntype
    item_ntype = data.item_ntype

    batch_size = 32
    random_walk_length = 2
    random_walk_restart_prob = 0.5
    num_random_walks = 10
    num_neighbors = 3
    num_layers = 2
    num_workers = 0
    hidden_dims = 16
    lr = 3e-5
    num_epochs = 30
    batches_per_epoch = 20000

    g = data[0]
    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, batch_size)
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, random_walk_length,
        random_walk_restart_prob, num_random_walks, num_neighbors,
        num_layers)
    collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=num_workers)
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=batch_size,
        collate_fn=collator.collate_test,
        num_workers=num_workers)
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(g, item_ntype, textset, hidden_dims, num_layers).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for batch_id in tqdm.trange(batches_per_epoch):
        pos_graph, neg_graph, blocks = next(dataloader_it)
        # Copy to GPU
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)

        loss = model(pos_graph, neg_graph, blocks).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("start training...")
    t0 = time.time()
    # For each batch of head-tail-negative triplets...
    for epoch_id in range(num_epochs):
        model.train()
        for batch_id in tqdm.trange(batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    t1 = time.time()

    return t1 - t0
