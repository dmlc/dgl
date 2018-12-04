"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV specialization.
"""
import argparse
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from data_utils import load_dataset




def message_sum(nodes):
        messages = fn.sum(msg='m', out='h')(nodes)
        #print(nodes.batch_size())
        messages['h'] = (nodes.data['h'] + messages['h'])
        return messages

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()

        self.activation = activation
        self.in_features = in_feats
        self.out_features = out_feats
        self.weight = Parameter(torch.FloatTensor(in_feats, out_feats))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.in_degree = None

    def forward(self, nodes):
        h = torch.mm(nodes.data['h'], self.weight)/self.in_degree.view(-1, 1)

        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)

        return {'h': h}

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

        # input layer
        self.layers = nn.ModuleList([NodeApplyModule(in_feats, n_hidden, activation)])

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NodeApplyModule(n_hidden, n_hidden, activation))

        # output layer
        self.layers.append(NodeApplyModule(n_hidden, n_classes))

    def forward(self, features):
        self.g.ndata['h'] = features

        for layer in self.layers:
            # apply dropout
            if layer == self.layers[0]:
                self.g.update_all(fn.copy_src(src='h', out='m'),
                                  message_sum,
                                  layer)
                continue

            if self.dropout:
                self.g.apply_nodes(func=lambda nodes: {'h': self.dropout(nodes.data['h'])})
            self.g.update_all(fn.copy_src(src='h', out='m'),
                              message_sum,
                              layer)
        return self.g.pop_n_repr('h')

def main(args):
    # load and preprocess dataset
    # Todo: adjacency normalization
    data = load_data(args)
    _, _, _, idx_train, idx_val, idx_test, graph = load_dataset()
    features = torch.FloatTensor(load_dataset()[1])
    labels = torch.LongTensor(load_dataset()[2])
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        #mask = mask.cuda()

    # create GCN model
    g = DGLGraph(load_dataset()[-1])
    in_degree_list = []

    for i in range(g.number_of_nodes()):
        in_degree_list.append(g.in_degree(i))

    if cuda:
        in_degree = torch.from_numpy(np.array(in_degree_list) + 1).type(torch.FloatTensor).cuda()
    else:
        in_degree = torch.from_numpy(np.array(in_degree_list) + 1).type(torch.FloatTensor)

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    for layer in model.layers:
        layer.in_degree = in_degree
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.item(), np.mean(dur), n_edges / np.mean(dur) / 1000))

    with torch.no_grad():
        model.eval()
        label = labels[idx_test]
        prediction = model(features)
        prediction = prediction.data.cpu().numpy()
        label = label.data.cpu().numpy()
        pred = np.argmax(prediction, axis=1)
        accur = np.where(pred[idx_test] == label, 1, 0)
        print("The accuracy: " + str(sum(accur)/len(accur)))

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
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    args = parser.parse_args()
    main(args)
