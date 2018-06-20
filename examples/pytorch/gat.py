import networkx as nx
from dgl.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import load_data, preprocess_features
import numpy as np

class NodeUpdateModule(nn.Module):
    def __init__(self, input_dim, num_hidden, aggregator, num_heads=3, act=None,
            attention_dropout=None, input_dropout=None, residual=False):
        super(NodeUpdateModule, self).__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.fc = nn.ModuleList(
                [nn.Linear(input_dim, num_hidden, bias=False)
                    for _ in range(num_heads)])
        self.attention = nn.ModuleList(
                [nn.Linear(num_hidden * 2, 1, bias=False) for _ in range(num_heads)])
        self.act = act
        self.attention_dropout = attention_dropout
        self.input_dropout = input_dropout
        self.aggregator = aggregator
        self.residual = residual

    def forward(self, node, msgs):
        hv = node['h']
        hu = torch.cat(msgs, dim=0)

        # number of neighbors, including itself
        n = len(msgs) + 1

        out = []
        for i in range(self.num_heads):
            hvv = hv
            huu = hu
            if self.input_dropout is not None:
                hvv = F.dropout(hvv, self.input_dropout)
                huu = F.dropout(huu, self.input_dropout)
            # calc W*hself and W*hneigh
            hvv = self.fc[i](hv)
            huu = self.fc[i](hu)
            # concat itself with neighbors to make self-attention
            huu = torch.cat((hvv, huu), dim=0)
            # calculate W*hself||W*hneigh
            h = torch.cat((hvv.expand(n, -1), huu), dim=1)
            a = F.leaky_relu(self.attention[i](h))
            a = F.softmax(a, dim=0)
            if self.attention_dropout is not None:
                a = F.dropout(a, self.attention_dropout)
            if self.input_dropout is not None:
                hvv = F.dropout(hvv, self.input_dropout)
            h = torch.sum(a * hvv, 0, keepdim=True)
            # add residual connection
            if self.residual:
                h += hvv
            if self.act is not None:
                h = self.act(h)
            out.append(h)

        # aggregate multi-head results
        h = self.aggregator(out)
        return {'h': h}


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, num_heads,
            activation, attention_dropout, input_dropout, use_residual=False):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        # update layers
        aggregator = lambda x: torch.cat(x, 1)
        for i in range(num_layers):
            if i == 0:
                last_dim = in_dim
                residual = False
            else:
                last_dim = num_hidden * num_heads # because of concat heads
                residual = use_residual
            self.layers.append(
                    NodeUpdateModule(last_dim, num_hidden, aggregator, num_heads,
                        activation, attention_dropout, input_dropout, residual))
        # projection layer
        # FIXME: does pytorch has something similar to tf.add_n which sum over a list?
        aggregator = lambda x: reduce(lambda a, b: a+b, x)
        self.layers.append(NodeUpdateModule(num_hidden * 3, num_classes, aggregator,
            1, None, attention_dropout, input_dropout, False))

    def forward(self, g):
        g.register_message_func(lambda src, dst, edge: src['h'])
        for layer in self.layers:
            g.register_update_func(layer)
            g.update_all()
        logits = [g.node[n]['h'] for n in g.nodes()]
        logits = torch.cat(logits, dim=0)
        return logits


def main(args):
    # dropout parameters
    input_dropout = 0.2
    attention_dropout = 0.2

    # load and preprocess dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    features = preprocess_features(features)

    # initialize graph
    g = DGLGraph(adj)

    # create model
    model = GAT(args.num_layers,
                features.shape[1],
                args.num_hidden,
                y_train.shape[1],
                args.num_heads,
                F.elu,
                attention_dropout,
                input_dropout,
                args.residual)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # convert labels and masks to tensor
    labels = torch.FloatTensor(y_train)
    mask = torch.FloatTensor(train_mask.astype(np.float32))

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
        loss = torch.mean(logp * labels * mask.view(-1, 1))
        print("epoch {} loss: {}".format(epoch, loss.item()))

        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset name")
    parser.add_argument("--epochs", type=int, default=10,
            help="training epoch")
    parser.add_argument("--num-heads", type=int, default=3,
            help="number of attentional heads to use")
    parser.add_argument("--num-layers", type=int, default=1,
            help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
            help="size of hidden units")
    parser.add_argument("--residual", action="store_true",
            help="use residual connection")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    args = parser.parse_args()
    print(args)

    main(args)

