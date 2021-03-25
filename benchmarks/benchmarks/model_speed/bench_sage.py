import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

from .. import utils


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes,
                           aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


@utils.benchmark('time')
@utils.parametrize('data', ['cora', 'pubmed'])
def track_time(data):
    dataset = utils.process_data(data)
    device = utils.get_bench_device()

    g = dataset[0].to(device)

    num_runs = 3
    num_epochs = 200

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    in_feats = features.shape[1]
    n_classes = dataset.num_classes

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    epoch_times = []

    for run in range(num_runs):
        # create model
        model = GraphSAGE(in_feats, 16, n_classes, 1, F.relu, 0.5, 'gcn')
        loss_fcn = torch.nn.CrossEntropyLoss()

        model = model.to(device)
        model.train()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1e-2,
                                     weight_decay=5e-4)

        # dry run
        for epoch in range(10):
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # timing
        for epoch in range(num_epochs):
            t0 = time.time()

            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()

            epoch_times.append(t1 - t0)

    avg_epoch_time = np.mean(epoch_times)
    std_epoch_time = np.std(epoch_times)

    std_const = 1.5
    low_boundary = avg_epoch_time - std_epoch_time * std_const
    high_boundary = avg_epoch_time + std_epoch_time * std_const

    valid_epoch_times = np.array(epoch_times)[(
        epoch_times >= low_boundary) & (epoch_times <= high_boundary)]
    avg_valid_epoch_time = np.mean(valid_epoch_times)

    # TODO: delete logging for final version
    print(f'Number of epoch times: {len(epoch_times)}')
    print(f'Number of valid epoch times: {len(valid_epoch_times)}')
    print(f'Avg epoch times: {avg_epoch_time}')
    print(f'Avg valid epoch times: {avg_valid_epoch_time}')

    return avg_valid_epoch_time
