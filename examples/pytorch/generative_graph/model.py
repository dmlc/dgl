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
    def __init__(self, node_num_hidden, graph_num_hidden, T, num_MLP_layers=1, loss_func=None, dropout=0.0, use_cuda=False):
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
        self.use_cuda = use_cuda

    def decide_add_node(self, hGs):
        h = self.fan(hGs)
        p = F.softmax(h, dim=1)
        # calc loss
        self.loss += self.loss_func(p, self.labels[self.step], self.masks[self.step])

    def decide_add_edge(self, batched_graph, hGs):
        hvs = batched_graph.get_n_repr((self.sample_node_curr_idx - 1).tolist())['h']
        h = self.fae(torch.cat((hGs, hvs), dim=1))
        p = torch.sigmoid(h)
        p = torch.cat([1 - p, p], dim=1)
        self.loss += self.loss_func(p, self.labels[self.step], self.masks[self.step])

    def select_node_to_add_edge(self, batched_graph, indices):
        node_indices = self.sample_node_curr_idx[indices].tolist()
        node_start = self.sample_node_start_idx[indices].tolist()
        node_repr = batched_graph.get_n_repr()['h']
        for i, j, idx in zip(node_start, node_indices, indices):
            hu = node_repr.narrow(0, i, j-i)
            hv = node_repr.narrow(0, j-1, 1)
            huv = torch.cat((hu, hv.expand(j-i, -1)), dim=1)
            s = F.softmax(self.fs(huv), dim=0).view(1, -1)
            dst = self.node_select[self.step][idx].view(-1)
            self.loss += self.loss_func(s, dst)

    def update_graph_repr(self, batched_graph, hGs, indices, indices_tensor):
        start = self.sample_node_start_idx[indices].tolist()
        stop = self.sample_node_curr_idx[indices].tolist()
        node_repr = batched_graph.get_n_repr()['h']
        graph_repr = self.graph_project(node_repr)
        new_hGs = []
        for i, j in zip(start, stop):
            h = graph_repr.narrow(0, i, j-i)
            hG = torch.sum(h, 0, keepdim=True)
            new_hGs.append(hG)
        new_hGs = torch.cat(new_hGs, dim=0)
        return hGs.index_copy(0, indices_tensor, new_hGs)

    def propagate(self, batched_graph, indices):
        edge_src = [self.sample_edge_src[idx][0: self.sample_edge_count[idx]] for idx in indices]
        edge_dst = [self.sample_edge_dst[idx][0: self.sample_edge_count[idx]] for idx in indices]
        u = np.concatenate(edge_src).tolist()
        v = np.concatenate(edge_dst).tolist()
        for gcn in self.gcn:
            gcn.forward(batched_graph, u, v, attribute='h')

    def forward(self, training=False, ground_truth=None):
        if not training:
            raise NotImplementedError("inference is not implemented yet")

        assert(ground_truth is not None)
        signals, (batched_graph, self.sample_edge_src, self.sample_edge_dst) = ground_truth
        nsteps, self.labels, self.node_select, self.masks, active_step, label1_set, label1_set_tensor = signals
        # init loss
        self.loss = 0

        batch_size = len(self.sample_edge_src)
        # initial node repr for each sample
        hVs = torch.zeros(len(batched_graph), self.node_num_hidden)
        # FIXME: what's the initial grpah repr for empty graph?
        hGs = torch.zeros(batch_size, self.graph_num_hidden)

        if self.use_cuda:
            hVs = hVs.cuda()
            hGs = hGs.cuda()
        batched_graph.set_n_repr({'h': hVs})

        self.sample_node_start_idx = batched_graph.query_node_start_offset()
        self.sample_node_curr_idx = self.sample_node_start_idx.copy()
        self.sample_edge_count = np.zeros(batch_size, dtype=int)

        self.step = 0
        while self.step < nsteps:
            if self.step % 2 == 0: # add node step
                if active_step[self.step]:
                    # decide whether to add node
                    self.decide_add_node(hGs)

                    # calculate initial state for new node
                    hvs = self.finit(hGs)

                    # add node
                    update = label1_set[self.step]
                    if len(update) > 0:
                        hvs = torch.index_select(hvs, 0, label1_set_tensor[self.step])
                        scatter_indices = self.sample_node_curr_idx[update]
                        batched_graph.set_n_repr({'h': hvs}, scatter_indices.tolist())
                        self.sample_node_curr_idx[update] += 1

                        # get new graph repr
                        hGs = self.update_graph_repr(batched_graph, hGs, update, label1_set_tensor[self.step])
                else:
                    # all samples are masked
                    pass

            else: # add edge step

                # decide whether to add edge, which edge to add
                # and also add edge
                self.decide_add_edge(batched_graph, hGs)

                # propagate
                to_add_edge = label1_set[self.step]
                if len(to_add_edge) > 0:
                    # at least one graph needs update
                    self.select_node_to_add_edge(batched_graph, to_add_edge)
                    # update edge count for each sample
                    self.sample_edge_count[to_add_edge] += 2 # undirected graph

                    # perform gcn propagation
                    self.propagate(batched_graph, to_add_edge)

                    # get new graph repr
                    hGs = self.update_graph_repr(batched_graph, hGs, label1_set[self.step], label1_set_tensor[self.step])

            self.step += 1


def main(args):

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        use_cuda = True
    else:
        use_cuda = False


    def masked_cross_entropy(x, label, mask=None):
        # x: propability tensor, i.e. after softmax
        x = torch.log(x)
        if mask is not None:
            x = x[mask]
            label = label[mask]
        return F.nll_loss(x, label)

    model = DGMG(args.n_hidden_node, args.n_hidden_graph, args.n_layers,
                 loss_func=masked_cross_entropy, dropout=args.dropout, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for ep in range(args.n_epochs):
        print("epoch: {}".format(ep))
        for idx, ground_truth in enumerate(DataLoader(args.dataset, args.batch_size)):
            if use_cuda:
                count, label, node_list, mask, active, label1, label1_tensor = ground_truth[0]
                label, node_list, mask, label1_tensor = move2cuda((label, node_list, mask, label1_tensor))
                ground_truth[0] = (count, label, node_list, mask, active, label1, label1_tensor)

            optimizer.zero_grad()
            # create new empty graphs
            start = time.time()
            model.forward(True, ground_truth)
            end = time.time()
            elapsed("model forward", start, end)
            start = time.time()
            model.loss.backward()
            optimizer.step()
            end = time.time()
            elapsed("model backward", start, end)
            print("iter {}: loss {}".format(idx, model.loss.item()))


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
