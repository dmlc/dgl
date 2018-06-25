import sys
sys.path.append("..")
from gcn import GCN
from dgl.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.node_num_hidden = node_num_hidden
        self.graph_num_hidden = graph_num_hidden
        # use GCN as a simple propagation model
        self.gcn = GCN(node_num_hidden, node_num_hidden, node_num_hidden, T, F.relu, output_projection=False)
        # project node repr to graph repr (higher dimension)
        self.graph_project = nn.Linear(node_num_hidden, graph_num_hidden)
        # add node
        self.fan = MLP(graph_num_hidden, 2, num_MLP_layers)
        # add edge
        self.fae = MLP(graph_num_hidden + node_num_hidden, 2, num_MLP_layers)
        # select node to add edge
        self.fs = MLP(node_num_hidden * 2, 1, num_MLP_layers)
        # init node state
        self.finit = MLP(graph_num_hidden, node_num_hidden, num_MLP_layers)

        self.train = False
        self.loss_func = loss_func

    # gcn propagate
    def propagate(self, g):
        if len(g) > 0:
            self.gcn.forward(g)

    def readout(self, g):
        hidden = self.gcn.readout(g)
        hG = torch.sum(self.graph_project(hidden), 0, keepdim=True)
        return hG

    # peek the correct choice
    def peek_ground_truth(self):
        return self.ordering[self.head]

    def set_ground_truth(self, ordering):
        self.ordering = ordering
        if ordering is None:
            # inference
            self.train = False
            self.node_count = 0
        else:
            # training
            self.ordering += [-1] # add a stop sign
            self.train = True
            self.head = 0
            self.loss = 0

    def decide_add_node(self, hG):
        h = self.fan(hG)
        if self.train:
            p = F.softmax(h, dim=1)
            # get ground truth
            ground_truth = self.peek_ground_truth()
            assert(isinstance(ground_truth, int))
            if ground_truth >= 0:
                label = 1 # keep adding node
            else:
                label = 0 # see stop (-1)
            # calc loss
            self.loss += self.loss_func(p, label)
            return label
        else:
            _, idx = torch.max(h)
            return idx

    def decide_add_edge(self, hG, hv):
        h = self.fae(torch.cat((hG, hv), dim=1))
        p = F.sigmoid(h)
        if self.train:
            # get ground truth
            ground_truth = self.peek_ground_truth()
            if isinstance(ground_truth, tuple):
                label = 1 # add edge
            else:
                label = 0 # add node
            # calc loss
            self.loss += self.loss_func(p, label)
            return label
        else:
            _, idx = torch.max(p)
            return idx

    def select_node_for_edge(self, g, hv):
        h = [g.node[n]['h'] for n in g.nodes()]
        hu = h[:-1]
        h = torch.cat((torch.cat(hu, dim=0), hv.expand(len(hu), -1)), dim=1)
        if self.train:
            s = F.softmax(self.fs(h), dim=0).view(1, -1)
            # get ground truth (src, dst)
            ground_truth = self.peek_ground_truth()
            assert(isinstance(ground_truth, tuple))
            # assuming new node is src node
            label = ground_truth[1]
            # calc loss
            self.loss += self.loss_func(s, label)
            # increment head
            self.head += 1
            return label
        else:
            _, idx = torch.max(self.fs(h).view(1, -1))
            return indx

    def get_next_node_id(self):
        if self.train:
            node_id = self.peek_ground_truth()
            # increment head
            self.head += 1
        else:
            node_id = self.node_count
            self.node_count += 1
        return node_id

    def init_node_state(self, hG):
        if hG is None:
            hG = torch.zeros((1, self.graph_num_hidden))
        return self.finit(hG)

    # g is an empty graph
    # ordering is a list of node(int) or edge(tuple) added to g (i.e. ground truth)
    # order set to None for inference
    def forward(self, g, ordering=None):
        self.set_ground_truth(ordering)
        hG = None # start with empty graph
        while len(g) == 0 or self.decide_add_node(hG):
            node_id = self.get_next_node_id()
            # add node
            g.add_node(node_id)
            # init node state
            hv = self.init_node_state(hG)
            g.node[node_id]['h'] = hv
            while len(g) > 1 and self.decide_add_edge(hG, hv):
                # add edges
                dst = self.select_node_for_edge(g, hv)
                g.add_edge(node_id, dst)
                # propagate
                self.propagate(g)
            # get new graph repr
            hG = self.readout(g)

def main():
    epoch = 10
    # number of hidden units
    node_num_hidden = 32
    graph_num_hidden = 64
    # number of rounds of propagation
    T = 2

    # graph
    # 0   1---3
    #  \ /
    #   2

    # ground truth
    ordering = [0, 1, 2, (2, 0), (2, 1), 3, (3, 1)]

    # loss function
    def loss_func(x, label):
        label = torch.LongTensor([[label]])
        y_onehot = torch.FloatTensor(x.shape).zero_()
        y_onehot.scatter_(1, label, 1)
        # return F.mse_loss(x, y_onehot)
        return -torch.mean(torch.log(x) * y_onehot)

    model = DGMG(node_num_hidden, graph_num_hidden, T, loss_func=loss_func)
    optimizer = torch.optim.Adam(model.parameters())
    for ep in range(epoch):
        print("epoch: {}".format(ep))
        optimizer.zero_grad()
        # create new empty graph
        g = DGLGraph()
        model.forward(g, ordering)
        print("loss: {}".format(model.loss.item()))
        model.loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
