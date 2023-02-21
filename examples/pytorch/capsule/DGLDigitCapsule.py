import dgl
import dgl.function as fn
import torch
from DGLRoutingLayer import DGLRoutingLayer
from torch import nn
from torch.nn import functional as F


class DGLDigitCapsuleLayer(nn.Module):
    def __init__(
        self,
        in_nodes_dim=8,
        in_nodes=1152,
        out_nodes=10,
        out_nodes_dim=16,
        device="cpu",
    ):
        super(DGLDigitCapsuleLayer, self).__init__()
        self.device = device
        self.in_nodes_dim, self.out_nodes_dim = in_nodes_dim, out_nodes_dim
        self.in_nodes, self.out_nodes = in_nodes, out_nodes
        self.weight = nn.Parameter(
            torch.randn(in_nodes, out_nodes, out_nodes_dim, in_nodes_dim)
        )

    def forward(self, x):
        self.batch_size = x.size(0)
        u_hat = self.compute_uhat(x)
        routing = DGLRoutingLayer(
            self.in_nodes,
            self.out_nodes,
            self.out_nodes_dim,
            batch_size=self.batch_size,
            device=self.device,
        )
        routing(u_hat, routing_num=3)
        out_nodes_feature = routing.g.nodes[routing.out_indx].data["v"]
        # shape transformation is for further classification
        return (
            out_nodes_feature.transpose(0, 1)
            .unsqueeze(1)
            .unsqueeze(4)
            .squeeze(1)
        )

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
        return u_hat.view(-1, self.batch_size, self.out_nodes_dim)
