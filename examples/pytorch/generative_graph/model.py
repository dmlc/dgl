import dgl
from dgl.nn import GCN
from dgl.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from util import DataLoader, pad_ground_truth

class MLP(nn.Module):
    def __init__(self, num_hidden, num_classes, num_layers):
        super(MLP, self).__init__()
        layers = []
        # hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.Sigmoid())
        # output projection
        layers.append(nn.Linear(num_hidden, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DGMG(nn.Module):
    def __init__(self, g, node_num_hidden, graph_num_hidden, T, num_MLP_layers=1, loss_func=None, dropout=0.0, use_cuda=False):
        super(DGMG, self).__init__()
        # hidden size of node and graph
        self.g = g
        self.node_num_hidden = node_num_hidden
        self.graph_num_hidden = graph_num_hidden
        # use GCN as a simple propagation model
        self.gcn = GCN(g, node_num_hidden, node_num_hidden, None, T, F.relu, dropout, projection=False)
        # project node repr to graph repr (higher dimension)
        self.graph_project = nn.Linear(node_num_hidden, graph_num_hidden)
        # add node
        self.fan = MLP(graph_num_hidden, 2, num_MLP_layers)
        # add edge
        self.fae = MLP(graph_num_hidden + node_num_hidden, 1, num_MLP_layers)
        # select node to add edge
        self.fs = MLP(node_num_hidden * 2, 1, num_MLP_layers)
        # init node state
        self.finit = MLP(graph_num_hidden, node_num_hidden, num_MLP_layers)
        # loss function
        self.loss_func = loss_func
        # use gpu
        self.use_cuda = use_cuda
        # create 1-label and 0-label for training use
        self.label1 = torch.ones(1, dtype=torch.long)
        self.label0 = torch.zeros(1, dtype=torch.long)
        if self.use_cuda:
            self.label1 = self.label1.cuda()
            self.label0 = self.label0.cuda()

    def decide_add_node(self, hG, label=1):
        h = self.fan(hG)
        p = F.softmax(h, dim=1)
        # calc loss
        label = self.label1 if label == 1 else self.label0
        self.loss += self.loss_func(p, label)

    def decide_add_edge(self, hG, hV, label=1):
        n = len(self.g)
        hv = hV.narrow(0, n - 1, 1)
        h = self.fae(torch.cat((hG, hv), dim=1))
        p = F.sigmoid(h)
        p = torch.cat([1 - p, p], dim=1)
        if label == 1:
            self.loss += self.loss_func(p, self.label1)

            # select node to add edge
            hu = hV.narrow(0, 0, n - 1)
            huv = torch.cat((hu, hv.expand(n - 1, -1)), dim=1)
            s = F.softmax(self.fs(huv), dim=0).view(1, -1)
            dst = torch.LongTensor([self.ground_truth[self.step][0]])
            if self.use_cuda:
                dst = dst.cuda()
            self.loss += self.loss_func(s, dst)
        else:
            self.loss += self.loss_func(p, self.label0)

    def forward(self, training=False, ground_truth=None):
        if training:
            assert(ground_truth is not None)
            # record ground_truth ordering
            self.ground_truth = ground_truth
            # init loss
            self.loss = 0
        else:
            raise NotImplementedError("inference is not implemented yet")

        # init
        self.g.clear()
        hV = torch.zeros(0, self.node_num_hidden)
        # FIXME: what's the initial grpah repr for empty graph?
        hG = torch.zeros(1, self.graph_num_hidden)

        if self.use_cuda:
            hV = hV.cuda()
            hG = hG.cuda()

        # step count
        self.step = 0

        nsteps = len(self.ground_truth)

        while self.step < nsteps:
            assert(not isinstance(self.ground_truth[self.step], tuple)) # add nodes

            # decide whether to add node
            self.decide_add_node(hG, 1)

            # add node
            self.g.add_node(self.ground_truth[self.step])

            # calculate initial state for new node
            hv = self.finit(hG)
            hV = torch.cat((hV, hv), dim=0)

            # get new graph repr
            hG = torch.sum(self.graph_project(hV), 0, keepdim=True)

            self.step += 1

            # decide whether to add edges (for at least once)
            while self.step < nsteps and isinstance(self.ground_truth[self.step], tuple):

                # decide whether to add edge, and which edge to add
                self.decide_add_edge(hG, hV, 1)

                # add edge
                self.g.add_edge(*self.ground_truth[self.step])

                # propagate
                hV = self.gcn.forward(hV)

                # get new graph repr
                hG = torch.sum(self.graph_project(hV), 0, keepdim=True)

                self.step += 1

            # decide not to add edges
            self.decide_add_edge(hG, hV, 0)

        # decide not to add nodes any more
        self.decide_add_node(hG, 0)


def main(args):
    # graph
    # 0---1   2
    #  \  |  /
    #   \ | /
    #     3

    # ground truth
    # ground_truth = [0, 1, (0, 1), 2, 3, (0, 3), (1, 3), (2, 3)]

    g = DGLGraph()

    use_cuda = False
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        use_cuda = True
        g.set_device(dgl.gpu(args.gpu))

    #model = DGMG(node_num_hidden, graph_num_hidden, T, loss_func=masked_loss_func)
    model = DGMG(g, args.n_hidden_node, args.n_hidden_graph, args.n_layers,
                 loss_func=F.cross_entropy, dropout=args.dropout, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for ep in range(args.n_epochs):
        print("epoch: {}".format(ep))
        for idx, batch in enumerate(DataLoader(args.dataset, args.batch_size)):
            label, node_select, mask = pad_ground_truth(batch)
            if use_cuda:
                label = label.cuda()
                node_select = node_select.cuda()
                mask = mask.cuda()
            optimizer.zero_grad()
            # create new empty graphs
            model.forward(True, batch[0])
            print("iter {}: loss {}".format(idx, model.loss.item()))
            model.loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGMG')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=1,
            help="number of training epochs")
    parser.add_argument("--n-hidden-node", type=int, default=16,
            help="number of hidden DGMG node units")
    parser.add_argument("--n-hidden-graph", type=int, default=32,
            help="number of hidden DGMG graph units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--dataset", type=str, default='samples.p',
            help="dataset pickle file")
    parser.add_argument("--batch-size", type=int, default=2,
            help="batch size")
    args = parser.parse_args()
    print(args)

    main(args)
