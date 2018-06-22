import networkx as nx
from dgl.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import load_data, preprocess_features
import numpy as np

class NodeUpdateModule(nn.Module):
    def __init__(self, input_dim, output_dim, act=None, p=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = act
        self.p = p

    def forward(self, node, msgs):
        h = node['h']
        # (lingfan): how to write dropout, is the following correct?
        if self.p is not None:
            h = F.dropout(h, p=self.p)
        # aggregate messages
        for msg in msgs:
            h += msg
        h = self.linear(h)
        if self.act is not None:
            h = self.act(h)
        # (lingfan): Can user directly update node instead of using return statement?
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, input_dim, num_hidden, num_classes, num_layers, activation, dropout=None, output_projection=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # hidden layers
        last_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(
                    NodeUpdateModule(last_dim, num_hidden, act=activation, p=dropout))
            last_dim = num_hidden
        # output layer
        if output_projection:
            self.layers.append(NodeUpdateModule(num_hidden, num_classes, p=dropout))

    def forward(self, g):
        g.register_message_func(lambda src, dst, edge: src['h'])
        for layer in self.layers:
            g.register_update_func(layer)
            g.update_all()
        logits = [g.node[n]['h'] for n in g.nodes()]
        return torch.cat(logits, dim=0)


def main(args):
    # load and preprocess dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    features = preprocess_features(features)

    # initialize graph
    g = DGLGraph(adj)

    # create GCN model
    model = GCN(features.shape[1],
                args.num_hidden,
                y_train.shape[1],
                args.num_layers,
                F.relu,
                args.dropout)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # convert labels and masks to tensor
    labels = torch.FloatTensor(y_train)
    mask = torch.FloatTensor(train_mask.astype(np.float32))
    n_train = torch.sum(mask)

    for epoch in range(args.epochs):
        # reset grad
        optimizer.zero_grad()

        # reset graph states
        for n in g.nodes():
            g.node[n]['h'] = torch.FloatTensor(features[n].toarray())

        # forward
        logits = model.forward(g)

        # masked cross entropy loss
        # TODO: (lingfan) use gather to speed up
        logp = F.log_softmax(logits, 1)
        loss = -torch.sum(logp * labels * mask.view(-1, 1)) / n_train
        print("epoch {} loss: {}".format(epoch, loss.item()))

        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset name")
    parser.add_argument("--num-layers", type=int, default=1,
            help="number of gcn layers")
    parser.add_argument("--num-hidden", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=10,
            help="training epoch")
    parser.add_argument("--dropout", type=float, default=None,
            help="dropout probability")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    args = parser.parse_args()
    print(args)

    main(args)

