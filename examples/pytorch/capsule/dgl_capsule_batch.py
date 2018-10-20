import dgl
import torch
import torch.nn.functional as F
from torch import nn

class DGLBatchCapsuleLayer(nn.Module):
    def __init__(self, in_unit, in_channel, num_unit, unit_size, num_routing, cuda_enabled):
        super(DGLBatchCapsuleLayer, self).__init__()
        self.device = "cuda" if cuda_enabled else "cpu"
        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.num_routing = num_routing
        self.unit_size = unit_size
        self.weight = nn.Parameter(torch.randn(in_channel, num_unit, unit_size, in_unit))
        self.g = self.construct_graph()

    def construct_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.in_channel + self.num_unit)
        self.in_channel_nodes = list(range(self.in_channel))
        self.capsule_nodes = list(range(self.in_channel, self.in_channel + self.num_unit))
        u, v = [], []
        for i in self.in_channel_nodes:
            for j in self.capsule_nodes:
                u.append(i)
                v.append(j)
        g.add_edges(u, v)
        return g

    def forward(self, x):

        self.batch_size = x.size(0)

        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
        W = self.weight.expand(self.batch_size, *self.weight.size())

        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()

        node_features = torch.zeros(self.in_channel + self.num_unit, self.batch_size, self.unit_size).to(self.device)

        b_ij = torch.zeros(self.in_channel, self.num_unit).to(self.device)

        self.g.set_e_repr({'b_ij': b_ij.view(-1)})
        self.g.set_n_repr({'h': node_features})
        self.g.set_e_repr({'u_hat': u_hat.view(-1, self.batch_size, self.unit_size)})

        for i in range(self.num_routing):
            self.g.update_all(self.capsule_msg, self.capsule_reduce, self.capsule_update)
            self.g.update_edge(edge_func=self.update_edge)

        capsule_node_feature = self.g.get_n_repr()['h'][self.in_channel:self.in_channel + self.num_unit]
        return capsule_node_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)

    def update_edge(self, u, v, edge):
        return {'b_ij': edge['b_ij'] + (v['h'] * edge['u_hat']).mean(dim=1).sum(dim=1)}

    @staticmethod
    def capsule_msg(src, edge):
        return {'b_ij': edge['b_ij'], 'h': src['h'], 'u_hat': edge['u_hat']}

    @staticmethod
    def capsule_reduce(node, msg):
        b_ij_c, u_hat = msg['b_ij'], msg['u_hat']
        c_i = F.softmax(b_ij_c, dim=0)
        s_j = (c_i.unsqueeze(2).unsqueeze(3) * u_hat).sum(dim=1)
        return {'h': s_j}

    def capsule_update(self, msg):
        v_j = self.squash(msg['h'])
        return {'h': v_j}

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s
