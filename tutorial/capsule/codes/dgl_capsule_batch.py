import dgl
import torch
import torch.nn.functional as F
from torch import nn

from capsule_layer import CapsuleLayer
# import main
from utils import writer, step

# global_step = main.global_step
device = "cuda" if torch.cuda.is_available() else "cpu"


class DGLFeature():
    """
    To wrap different shape of representation tensor into the same shape
    """

    def __init__(self, tensor, pad_to):
        # self.tensor = tensor
        self.node_num = tensor.size(0)
        self.flat_tensor = tensor.contiguous().view(self.node_num, -1)
        self.node_feature_dim = self.flat_tensor.size(1)
        self.flat_pad_tensor = F.pad(self.flat_tensor, (0, pad_to - self.flat_tensor.size(1)))
        self.shape = tensor.shape

    @property
    def tensor(self):
        """
        :return: Tensor with original shape
        """
        return self.flat_tensor.index_select(1, torch.arange(0, self.node_feature_dim).to(device)).view(self.shape)

    @property
    def padded_tensor(self):
        """
        :return: Flatted and padded Tensor
        """
        return self.flat_pad_tensor


class DGLBatchCapsuleLayer(CapsuleLayer):
    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled):
        super(DGLBatchCapsuleLayer, self).__init__(in_unit, in_channel, num_unit, unit_size, use_routing,
                                                   num_routing, cuda_enabled)
        self.unit_size = unit_size
        self.weight = nn.Parameter(torch.randn(in_channel, num_unit, unit_size, in_unit))

    def routing(self, x):

        self.batch_size = x.size(0)

        self.g = dgl.DGLGraph()

        self.g.add_nodes_from([i for i in range(self.in_channel)])
        self.g.add_nodes_from([i + self.in_channel for i in range(self.num_unit)])
        for i in range(self.in_channel):
            for j in range(self.num_unit):
                index_j = j + self.in_channel
                self.g.add_edge(i, index_j)

        self.edge_features = torch.zeros(self.in_channel, self.num_unit).to('cuda')

        x_ = x.transpose(0, 2)
        x_ = DGLFeature(x_, self.batch_size * self.unit_size)

        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
        W = torch.cat([self.weight.unsqueeze(0)] * self.batch_size, dim=0)
        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()

        self.node_feature = DGLFeature(torch.zeros(self.num_unit, self.batch_size, self.unit_size).to('cuda'),
                                       self.batch_size * self.unit_size)
        nf = torch.cat([x_.padded_tensor, self.node_feature.padded_tensor], dim=0)

        self.g.set_e_repr({'b_ij': self.edge_features.view(-1)})
        self.g.set_n_repr({'h': nf})
        self.g.set_e_repr({'u_hat': u_hat.view(-1, self.batch_size, self.unit_size)})

        for i in range(self.num_routing):
            self.i = i
            self.g.update_all(self.capsule_msg, self.capsule_reduce,
                              lambda x: {'h': DGLFeature(x['h'], self.batch_size * self.unit_size).padded_tensor},
                              batchable=True)
            self.g.update_edge(dgl.base.ALL, dgl.base.ALL, self.update_edge, batchable=True)

        self.node_feature = self.g.get_n_repr()['h'] \
            .index_select(0, torch.arange(self.in_channel, self.in_channel + self.num_unit).to(device)) \
            .view(self.num_unit, self.batch_size, self.unit_size)
        return self.node_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)

    def update_edge(self, u, v, edge):
        return {
            'b_ij': edge['b_ij'] + (v['h'].view(-1, self.batch_size, self.unit_size) * edge['u_hat']).mean(dim=1).sum(
                dim=1)}

    @staticmethod
    def capsule_msg(src, edge):
        return {'b_ij': edge['b_ij'], 'h': src['h'], 'u_hat': edge['u_hat']}

    def capsule_reduce(self, node, msg):

        b_ij_c, h_c, u_hat_c = msg['b_ij'], msg['h'], msg['u_hat']
        u_hat = u_hat_c
        c_i = F.softmax(b_ij_c, dim=0)
        writer.add_histogram(f"c_i{self.i}", c_i, step['step'])
        s_j = (c_i.unsqueeze(2).unsqueeze(3) * u_hat).sum(dim=1)
        v_j = self.squash(s_j)
        return {'h': v_j.view(-1, self.batch_size * self.unit_size)}

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s
