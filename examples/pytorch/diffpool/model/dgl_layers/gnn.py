import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import block_diag

from model.loss import EntropyLoss
from ..model_utils import masked_softmax

from .aggregator import LSTMAggregator, MaxPoolAggregator, MeanAggregator
from .bundler import Bundler


class GraphSageLayer(nn.Module):
    """
    GraphSage layer in Inductive learning paper by hamilton
    Here, graphsage layer is a reduced function in DGL framework
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        activation,
        dropout,
        aggregator_type,
        bn=False,
        bias=True,
    ):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.bundler = Bundler(
            in_feats, out_feats, activation, dropout, bias=bias
        )
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(
                in_feats, in_feats, activation, bias
            )
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata["h"] = h
        if self.use_bn and not hasattr(self, "bn"):
            device = h.device
            self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
        g.update_all(fn.copy_u(u="h", out="m"), self.aggregator, self.bundler)
        if self.use_bn:
            h = self.bn(h)
        h = g.ndata.pop("h")
        return h


class GraphSage(nn.Module):
    """
    Grahpsage network that concatenate several graphsage layer
    """

    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
        aggregator_type,
    ):
        super(GraphSage, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphSageLayer(
                in_feats, n_hidden, activation, dropout, aggregator_type
            )
        )
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(
                GraphSageLayer(
                    n_hidden, n_hidden, activation, dropout, aggregator_type
                )
            )
        # output layer
        self.layers.append(
            GraphSageLayer(n_hidden, n_classes, None, dropout, aggregator_type)
        )

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class DiffPoolBatchedGraphLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        assign_dim,
        output_feat_dim,
        activation,
        dropout,
        aggregator_type,
        link_pred,
    ):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = GraphSageLayer(
            input_dim, output_feat_dim, activation, dropout, aggregator_type
        )
        self.pool_gc = GraphSageLayer(
            input_dim, assign_dim, activation, dropout, aggregator_type
        )
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):
        feat = self.feat_gc(
            g, h
        )  # size = (sum_N, F_out), sum_N is num of nodes in this batch
        device = feat.device
        assign_tensor = self.pool_gc(
            g, h
        )  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(
            *assign_tensor
        )  # size = (sum_N, batch_size * N_a)

        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adj_external(transpose=True, ctx=device)
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        if self.link_pred:
            current_lp_loss = torch.norm(
                adj.to_dense() - torch.mm(assign_tensor, torch.t(assign_tensor))
            ) / np.power(g.num_nodes(), 2)
            self.loss_log["LinkPredLoss"] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)

        return adj_new, h
