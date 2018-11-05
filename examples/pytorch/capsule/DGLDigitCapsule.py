import dgl
import torch
from torch import nn
from torch.nn import functional as F


class DGLDigitCapsuleLayer(nn.Module):
    def __init__(self, in_nodes_dim=8, in_nodes=1152, out_nodes=10, out_nodes_dim=16, device='cpu'):
        super(DGLDigitCapsuleLayer, self).__init__()
        self.device = device
        self.in_nodes_dim, self.out_nodes_dim = in_nodes_dim, out_nodes_dim
        self.in_nodes, self.out_nodes = in_nodes, out_nodes
        self.weight = nn.Parameter(torch.randn(in_nodes, out_nodes, out_nodes_dim, in_nodes_dim))
        self.g = self.construct_graph()

        def cap_message(edge):
            return {'b_ij': edge.data['b_ij'], 'h': edge.src['h'], 'u_hat': edge.data['u_hat']}
        self.g.register_message_func(cap_message)

        def cap_reduce(node):
            b_ij_c, u_hat = node.mailbox['b_ij'], node.mailbox['u_hat']
            # TODO: group_apply_edge
            c_i = F.softmax(b_ij_c, dim=0)
            s_j = (c_i.unsqueeze(2).unsqueeze(3) * u_hat).sum(dim=1)
            return {'h': s_j}
        self.g.register_reduce_func(cap_reduce)

        def cap_apply(node):
            v_j = squash(node.data['h'])
            return {'h': v_j}
        self.g.register_apply_node_func(cap_apply)

        def cap_apply_edges(edge):
            return {'b_ij': edge.data['b_ij'] + (edge.dst['h'] * edge.data['u_hat']).mean(dim=1).sum(dim=1)}
        self.g.register_apply_edge_func(cap_apply_edges)

    def forward(self, x):
        self.batch_size = x.size(0)
        u_hat = self.compute_uhat(x)
        self.initialize_nodes_and_edges_features(u_hat)
        self.routing(3)
        return self.get_out_nodes_repr()

    def routing(self, r):
        for i in range(r):
            self.g.update_all()
            self.g.apply_edges()

    def construct_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.in_nodes + self.out_nodes)
        in_nodes_idx = list(range(self.in_nodes))
        out_nodes_idx = list(range(self.in_nodes, self.in_nodes + self.out_nodes))
        u, v = [], []
        for i in in_nodes_idx:
            for j in out_nodes_idx:
                u.append(i)
                v.append(j)
        g.add_edges(u, v)
        return g

    def compute_uhat(self, x):
        # x is the input vextor with shape [batch_size, in_nodes_dim, in_nodes]
        # Transpose x to [batch_size, in_nodes, in_nodes_dim]
        x = x.transpose(1, 2)
        # Expand x to [batch_size, in_nodes, out_nodes, in_nodes_dim, 1]
        x = torch.stack([x] * self.out_nodes, dim=2).unsqueeze(4)
        # Expand W from [in_nodes, out_nodes, in_nodes_dim, out_nodes_dim]
        # to [batch_size, in_nodes, out_nodes, out_nodes_dim, in_nodes_dim]
        W = self.weight.expand(self.batch_size, *self.weight.size())
        # u_hat's shape is [in_nodes, out_nodes, batch_size, out_nodes_dim]
        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()
        return u_hat

    def initialize_nodes_and_edges_features(self, u_hat):
        b_ij = torch.zeros(self.in_nodes, self.out_nodes).to(self.device)
        self.g.set_e_repr({'b_ij': b_ij.view(-1)})
        self.g.set_e_repr({'u_hat': u_hat.view(-1, self.batch_size, self.out_nodes_dim)})

        # Initialize all node features as zero
        node_features = torch.zeros(self.in_nodes + self.out_nodes, self.batch_size,
                                    self.out_nodes_dim).to(self.device)
        self.g.set_n_repr({'h': node_features})

    def get_out_nodes_repr(self):
        out_nodes_feature = self.g.get_n_repr()['h'][
                            self.in_nodes:self.in_nodes + self.out_nodes]
        # shape transformation is for further classification
        return out_nodes_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)


def squash(s, dim=2):
    sq = torch.sum(s ** 2, dim=dim, keepdim=True)
    s_std = torch.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_std)
    return s
