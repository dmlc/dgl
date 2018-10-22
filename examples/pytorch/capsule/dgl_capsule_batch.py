import dgl
import torch
import torch.nn.functional as F
from torch import nn


class DGLBatchCapsuleLayer(nn.Module):
    def __init__(self, input_capsule_dim, input_capsule_num, output_capsule_num, output_capsule_dim, num_routing,
                 cuda_enabled):
        super(DGLBatchCapsuleLayer, self).__init__()
        self.device = "cuda" if cuda_enabled else "cpu"
        self.input_capsule_dim = input_capsule_dim
        self.input_capsule_num = input_capsule_num
        self.output_capsule_dim = output_capsule_dim
        self.output_capsule_num = output_capsule_num
        self.num_routing = num_routing
        self.weight = nn.Parameter(
            torch.randn(input_capsule_num, output_capsule_num, output_capsule_dim, input_capsule_dim))
        self.g, self.pre_layer_nodes, self.this_layer_nodes = self.construct_graph()

    def construct_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.input_capsule_num + self.output_capsule_num)
        pre_layer_nodes = list(range(self.input_capsule_num))
        this_layer_nodes = list(range(self.input_capsule_num, self.input_capsule_num + self.output_capsule_num))
        u, v = [], []
        for i in pre_layer_nodes:
            for j in this_layer_nodes:
                u.append(i)
                v.append(j)
        g.add_edges(u, v)
        return g, pre_layer_nodes, this_layer_nodes

    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.output_capsule_num, dim=2).unsqueeze(4)
        W = self.weight.expand(self.batch_size, *self.weight.size())
        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()

        b_ij = torch.zeros(self.input_capsule_num, self.output_capsule_num).to(self.device)

        self.g.set_e_repr({'b_ij': b_ij.view(-1)})
        self.g.set_e_repr({'u_hat': u_hat.view(-1, self.batch_size, self.output_capsule_dim)})

        node_features = torch.zeros(self.input_capsule_num + self.output_capsule_num, self.batch_size,
                                    self.output_capsule_dim).to(self.device)
        self.g.set_n_repr({'h': node_features})

        for i in range(self.num_routing):
            self.g.update_all(self.capsule_msg, self.capsule_reduce, self.capsule_update)
            self.g.update_edge(edge_func=self.update_edge)

        this_layer_nodes_feature = self.g.get_n_repr()['h'][
                                   self.input_capsule_num:self.input_capsule_num + self.output_capsule_num]
        return this_layer_nodes_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)

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
