import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from original import CapsuleLayer


class DGLCapsuleLayer(CapsuleLayer):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing=True):
        super(DGLCapsuleLayer, self).__init__(in_units, in_channels, num_units, unit_size, use_routing=True)
        self.g = dgl.DGLGraph(nx.from_numpy_matrix(np.ones((10, 10))))
        self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        # self.node_features = nn.Parameter(torch.randn(()))

    def routing(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        self.u_hat = u_hat
        self.node_feature = u_hat.clone().detach().transpose(0, 2).transpose(1, 2)
        self.g.set_n_repr({'ft': self.node_feature})

        self.edge_features = torch.zeros(100, 1)
        self.g.set_e_repr({'b_ij': self.edge_features})

        self.g.update_all(self.capsule_msg, self.capsule_reduce, lambda x: x)
        self.g.update_all(self.capsule_msg, self.capsule_reduce, lambda x: x)
        self.g.update_all(self.capsule_msg, self.capsule_reduce, lambda x: x)

        self.edge_features = self.edge_features + torch.dot(self.u_hat, self.node_feature)

    @staticmethod
    def capsule_msg(src, edge):
        return {'b_ij': edge['weight'], 'h': src['ft']}

    def capsule_reduce(self, node, msg):
        b_ij, h = msg
        b_ij_c, h_c = torch.cat(b_ij, dim=1), torch.cat(h, dim=1)
        c_i = F.softmax(b_ij_c, dim=1)
        s_j = torch.dot(c_i, self.u_hat)
        v_j = self.squash(s_j)
