"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


def graphsage_msg(edge):
    msg = edge.src['h']
    return {'m': msg}


class Aggregator(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))  # (F,EF)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))  # (EF,1)
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        weight_stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-weight_stdv, weight_stdv)
        if self.bias is not None:
            bias_stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-bias_stdv, bias_stdv)

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node)  # (B, F)
        h = torch.mm(h, self.weight)  # (B, EF)
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}

    def concat(self, h, nei, nodes):
        print('no implementation !')
        pass


class MeanAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(g, in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes):
        degs = self.g.in_degrees(nodes.nodes()).float()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        return concatenate


class PoolingAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(g, in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes):
        nei = self.mlp(nei)
        concatenate = torch.cat((nei, h), 1)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))  # (F,EF)
            if dropout:
                self.dropout = nn.Dropout(p=dropout)
            else:
                self.dropout = 0.
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_feats))  # (EF)
            else:
                self.bias = None
            self.activation = activation
            self.reset_parameters()

        def reset_parameters(self):
            weight_stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-weight_stdv, weight_stdv)
            if self.bias is not None:
                bias_stdv = 1. / math.sqrt(self.bias.size(0))
                self.bias.data.uniform_(-bias_stdv, bias_stdv)

        def forward(self, nei):  # (B, N, F)
            if self.dropout:
                nei = self.dropout(nei)
            nei = torch.matmul(nei, self.weight)  # (B, N, EF)
            if self.bias is not None:
                nei = nei + self.bias
            if self.activation:
                nei = self.activation(nei)
            max_value = torch.max(nei, dim=1)[0]  # (B, EF)
            return max_value


class GraphSAGELayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GraphSAGELayer, self).__init__()
        self.g = g
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(g, in_feats, out_feats, activation, bias)
        else:
            self.aggregator = MeanAggregator(g, in_feats, out_feats, activation, bias)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata['h'] = h
        self.g.update_all(graphsage_msg, self.aggregator)
        h = self.g.ndata.pop('h')
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(g, in_feats, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    # graph preprocess and calculate normalization factor
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()
    # add self loop
    # g.add_edges(g.nodes(), g.nodes())
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GraphSAGE(g,
                      in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type
                      )

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
