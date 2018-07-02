import sys
sys.path.append("..")
from gcn import GCN
from dgl.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import pad_and_mask

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
    def __init__(self, node_num_hidden, graph_num_hidden, T, num_MLP_layers=1, loss_func=None):
        super(DGMG, self).__init__()
        # hidden size of node and graph
        self.node_num_hidden = node_num_hidden
        self.graph_num_hidden = graph_num_hidden
        # use GCN as a simple propagation model
        self.gcn = GCN(node_num_hidden, node_num_hidden, node_num_hidden, T, F.relu, output_projection=False)
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
        # (masked) loss function
        self.loss_func = loss_func


    # waiting for batched version for progagtion and readout
    def propagate(self, gs):
        for mask, g in zip(self.masks[self.step], gs):
            if mask == 1:
                # only propagate graphs that have new edge added
                self.gcn.forward(g)


    def readout(self, gs):
        to_concat = []
        for g in gs:
            hidden = self.gcn.readout(g)
            hG = torch.sum(self.graph_project(hidden), 0, keepdim=True)
            to_concat.append(hG)
        if len(to_concat) == 1:
            return to_concat[0]
        else:
            return torch.cat(to_concat, dim=0)


    def decide_add_node(self, hG, label=1):
        h = self.fan(hG)
        p = F.softmax(h, dim=1)
        # calc loss
        if label == 1:
            label = self.label1
            mask = self.masks_tensor[self.step]
        else:
            label = self.label0
            mask = None
        self.loss += self.loss_func(p, label, mask)


    def decide_add_edge(self, hG, hv, label=1):
        h = self.fae(torch.cat((hG, hv), dim=1))
        p = F.sigmoid(h)
        p = torch.cat([1 - p, p], dim=1)

        # calc loss
        if label == 1:
            label = self.label1
            mask = self.masks_tensor[self.step]
        else:
            label = self.label0
            mask = None
        self.loss += self.loss_func(p, label, mask)


    def select_node_for_edge(self, g, hv, src, dst):
        hu = [g.node[n]['h'] for n in g.nodes() if n != dst]
        h = torch.cat((torch.cat(hu, dim=0), hv.expand(len(hu), -1)), dim=1)
        s = F.softmax(self.fs(h), dim=0).view(1, -1)
        # calc loss
        self.loss += self.loss_func(s, src)


    def add_node_for_batch(self, gs, new_states):
        for idx, (g, mask, node) in enumerate(zip(gs,
                                                 self.masks[self.step],
                                                 self.selected[self.step])):
            if mask == 1:
                g.add_node(node)
                g.node[node]['h'] = new_states[idx:idx+1] # keep dim


    def add_edge_for_batch(self, gs, hv):
        for idx, (g, mask, src, dst) in enumerate(zip(gs,
                                                      self.masks[self.step],
                                                      self.selected[self.step],
                                                      self.selected[self.last_node])):
            if mask == 1:
                # select node to add edge
                self.select_node_for_edge(g, hv[idx: idx + 1], src, dst)
                # add ground truth edge
                g.add_edge(src, dst)

    def set_ground_truths(self, ground_truths):
        # init ground truth
        actions, masks, selected = ground_truths
        self.actions = actions
        self.selected = selected
        self.masks = masks
        self.masks_tensor = list(map(torch.FloatTensor, masks))

    def forward(self, gs, training=False):
        if not training:
            raise NotImplementedError("batching for inference is not implemented yet")

        # init loss
        self.loss = 0

        self.batch_size = len(gs)

        # create 1-label and 0-label for future use
        self.label1 = torch.ones(self.batch_size, 1, dtype=torch.long)
        self.label0 = torch.zeros(self.batch_size, 1, dtype=torch.long)

        # start with empty graph
        hG = torch.zeros(self.batch_size, self.graph_num_hidden)

        # step count
        self.step = 0

        nsteps = len(self.actions)

        while self.step < nsteps:
            assert(self.actions[self.step] == 0) # add nodes

            # decide whether to add node
            self.decide_add_node(hG, 1)

            # batched add node
            hv = self.finit(hG)
            self.add_node_for_batch(gs, hv)
            self.last_node = self.step

            self.step += 1

            # decide whether to add edges (for at least once)
            while self.step < nsteps and self.actions[self.step] == 1:

                # decide whether to add edge
                self.decide_add_edge(hG, hv, 1)

                # batched add edge
                self.add_edge_for_batch(gs, hv)

                # propagate
                self.propagate(gs)

                # get new graph repr
                hG = self.readout(gs)

                self.step += 1

            # decide to stop add edges
            self.decide_add_edge(hG, hv, 0)

        # decide to stop add nodes
        self.decide_add_node(hG, 0)


def main():
    epoch = 10

    # number of hidden units
    node_num_hidden = 4
    graph_num_hidden = 8

    # number of rounds of propagation
    T = 2

    # graph1
    # 0   1
    #  \ /
    #   2

    # graph2
    # 0---1   2
    #  \  |  /
    #   \ | /
    #     3

    # ground truth
    orderings = [
                 [0, 1, 2, (0, 2), (1, 2)],
                 [0, 1, (0, 1), 2, 3, (0, 3), (1, 3), (2, 3)]
                ]

    batch_size = len(orderings)

    # pad and generate mask for samples in the batch for batching
    ground_truths = pad_and_mask(orderings)

    # loss function
    def masked_loss_func(x, label, mask=None):
        if isinstance(label, int):
            label = torch.LongTensor([[label]])
        # create one-hot code
        y_onehot = torch.FloatTensor(x.shape).zero_()
        y_onehot.scatter_(1, label, 1)
        # return F.mse_loss(x, y_onehot)
        if mask is not None:
            return -torch.sum(torch.log(x) * y_onehot * mask.view(-1, 1)) / torch.sum(mask)
        else:
            return -torch.mean(torch.log(x) * y_onehot)

    model = DGMG(node_num_hidden, graph_num_hidden, T, loss_func=masked_loss_func)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # set ground truth for training
    model.set_ground_truths(ground_truths)

    # training loop
    for ep in range(epoch):
        print("epoch: {}".format(ep))
        optimizer.zero_grad()
        # create new empty graphs
        gs = [DGLGraph() for _ in range(batch_size)]
        model.forward(gs, training=True)
        print("loss: {}".format(model.loss.item()))
        model.loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
