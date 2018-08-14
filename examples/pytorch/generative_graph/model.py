import dgl
from dgl.graph import DGLGraph
from dgl.nn import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from util import DataLoader, elapsed
import time

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


def move2cuda(x):
    # recursively move a object to cuda
    if isinstance(x, torch.Tensor):
        # if Tensor, move directly
        return x.cuda()
    else:
        try:
            # iterable, recursively move each element
            x = [move2cuda(i) for i in x]
            return x
        except:
            # don't do anything for other types like basic types
            return x


class DGMG(nn.Module):
    def __init__(self, node_num_hidden, graph_num_hidden, T, num_MLP_layers=1, loss_func=None, dropout=0.0, cuda_device=-1):
        super(DGMG, self).__init__()
        # hidden size of node and graph
        self.node_num_hidden = node_num_hidden
        self.graph_num_hidden = graph_num_hidden
        # use GCN as a simple propagation model
        self.gcn = nn.ModuleList([GCN(node_num_hidden, node_num_hidden, F.relu, dropout) for _ in range(T)])
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
        self.cuda_device = cuda_device

    def decide_add_node(self, hGs):
        h = self.fan(hGs)
        p = F.softmax(h, dim=1)
        # calc loss
        self.loss += self.loss_func(p, self.labels[self.step], self.masks[self.step])

    def decide_add_edge(self, hGs, graph_list):
        hvs = [g.get_n_repr(len(g) - 1)['h'] for g in graph_list]
        h = self.fae(torch.cat((hGs, torch.cat(hvs, dim=0)), dim=1))
        p = F.sigmoid(h)
        p = torch.cat([1 - p, p], dim=1)
        self.loss += self.loss_func(p, self.labels[self.step], self.masks[self.step])

        # select node to add edge
        for idx, g in enumerate(graph_list):
            if self.labels[self.step][idx].item() == 1:
                n = len(g)
                hV = g.get_n_repr()['h']
                hu = hV.narrow(0, 0, n - 1)
                huv = torch.cat((hu, hvs[idx].expand(n - 1, -1)), dim=1)
                s = F.softmax(self.fs(huv), dim=0).view(1, -1)
                dst = self.node_select[self.step][idx].view(-1)
                self.loss += self.loss_func(s, dst)

                # add edge
                src = n - 1
                dst = dst.item()
                g.add_edge(src, dst)
                g.add_edge(dst, src)

    def update_graph_repr(self, hG, graph_list):
        new_hGs = []
        for idx, g in enumerate(graph_list):
            features = g.get_n_repr()['h']
            hG = torch.sum(self.graph_project(features), 0, keepdim=True)
            new_hGs.append(hG)
        return torch.cat(new_hGs, dim=0)

    def forward(self, training=False, batch_size=1, ground_truth=None):
        if training:
            assert(ground_truth is not None)
            # record ground_truth ordering
            if self.cuda_device >= 0:
                ground_truth = move2cuda(ground_truth)
            nsteps, self.labels, self.node_select, self.masks = ground_truth
            # init loss
            self.loss = 0
        else:
            raise NotImplementedError("inference is not implemented yet")

        # create empty graph for each sample
        graph_list = [DGLGraph() for _ in range(batch_size)]

        # initial node repr for each sample
        hVs = [torch.zeros(0, self.node_num_hidden) for _ in range(batch_size)]

        # initial graph repr for each sample, set to zero tensor
        # FIXME: what's the initial grpah repr for empty graph?
        hGs = torch.zeros(batch_size, self.graph_num_hidden)

        if self.cuda_device >= 0:
            hVs = move2cuda(hVs)
            hGs = move2cuda(hGs)
            for g in graph_list:
                g.set_device(dgl.gpu(self.cuda_device))

        self.step = 0
        while self.step < nsteps:
            if self.step % 2 == 0: # add node step

                if (self.masks[self.step] == 1).nonzero().nelement() > 0:
                    # decide whether to add node
                    self.decide_add_node(hGs)

                    # calculate initial state for new node
                    hvs = self.finit(hGs)

                    # add node
                    update = []
                    for idx, g in enumerate(graph_list):
                        if self.labels[self.step][idx].item() == 1:
                            if self.step > 0:
                                hV = g.pop_n_repr('h')
                            else:
                                hV = hVs[idx]
                            hV = torch.cat((hV, hvs[idx:idx+1]), dim=0)
                            g.add_node(len(g))
                            g.set_n_repr({'h': hV})

                    # get new graph repr
                    hGs = self.update_graph_repr(hGs, graph_list)
                else:
                    # all samples are masked
                    pass

            else: # add edge step

                # decide whether to add edge, which edge to add
                # and also add edge
                self.decide_add_edge(hGs, graph_list)

                # propagate
                to_update = (self.labels[self.step] == 1).nonzero()
                if to_update.nelement() > 0:
                    # at least one graph needs update
                    to_update = [graph_list[i] for i in to_update]
                    batched_graph = dgl.batch(to_update)
                    # FIXME: should dgl.batch() handle set_device?
                    if self.cuda_device >= 0:
                        batched_graph.set_device(dgl.gpu(self.cuda_device))
                    for gcn in self.gcn:
                        gcn.forward(batched_graph, attribute='h')
                    dgl.unbatch(batched_graph)

                    # get new graph repr
                    hGs = self.update_graph_repr(hGs, graph_list)

            self.step += 1


def main(args):

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        cuda_device = args.gpu
    else:
        cuda_device = -1


    def masked_cross_entropy(x, label, mask=None):
        # x: propability tensor, i.e. after softmax
        x = torch.log(x)
        if mask is not None:
            x = x[mask]
            label = label[mask]
        return F.nll_loss(x, label)

    model = DGMG(args.n_hidden_node, args.n_hidden_graph, args.n_layers,
                 loss_func=masked_cross_entropy, dropout=args.dropout, cuda_device=cuda_device)
    if cuda_device >= 0:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for ep in range(args.n_epochs):
        print("epoch: {}".format(ep))
        for idx, ground_truth in enumerate(DataLoader(args.dataset, args.batch_size)):
            optimizer.zero_grad()
            # create new empty graphs
            start = time.time()
            model.forward(True, args.batch_size, ground_truth)
            end = time.time()
            print("iter {}: loss {}".format(idx, model.loss.item()))
            elapsed("model forward", start, end)
            model.loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGMG')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden-node", type=int, default=16,
            help="number of hidden DGMG node units")
    parser.add_argument("--n-hidden-graph", type=int, default=32,
            help="number of hidden DGMG graph units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--dataset", type=str, default='samples.p',
            help="dataset pickle file")
    parser.add_argument("--batch-size", type=int, default=32,
            help="batch size")
    args = parser.parse_args()
    print(args)

    main(args)
