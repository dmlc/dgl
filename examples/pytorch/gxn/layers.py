from typing import Optional

import dgl

import torch
import torch.nn
from dgl import DGLGraph
from dgl.nn import GraphConv
from torch import Tensor


class GraphConvWithDropout(GraphConv):
    """
    A GraphConv followed by a Dropout.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        dropout=0.3,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GraphConvWithDropout, self).__init__(
            in_feats,
            out_feats,
            norm,
            weight,
            bias,
            activation,
            allow_zero_in_degree,
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def call(self, graph, feat, weight=None):
        feat = self.dropout(feat)
        return super(GraphConvWithDropout, self).call(graph, feat, weight)


class Discriminator(torch.nn.Module):
    """
    Description
    -----------
    A discriminator used to let the network to discrimate
    between positive (neighborhood of center node) and
    negative (any neighborhood in graph) samplings.

    Parameters
    ----------
    feat_dim : int
        The number of channels of node features.
    """

    def __init__(self, feat_dim: int):
        super(Discriminator, self).__init__()
        self.affine = torch.nn.Bilinear(feat_dim, feat_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)

    def forward(
        self,
        h_x: Tensor,
        h_pos: Tensor,
        h_neg: Tensor,
        bias_pos: Optional[Tensor] = None,
        bias_neg: Optional[Tensor] = None,
    ):
        """
        Parameters
        ----------
        h_x : torch.Tensor
            Node features, shape: :obj:`(num_nodes, feat_dim)`
        h_pos : torch.Tensor
            The node features of positive samples
            It has the same shape as :obj:`h_x`
        h_neg : torch.Tensor
            The node features of negative samples
            It has the same shape as :obj:`h_x`
        bias_pos : torch.Tensor
            Bias parameter vector for positive scores
            shape: :obj:`(num_nodes)`
        bias_neg : torch.Tensor
            Bias parameter vector for negative scores
            shape: :obj:`(num_nodes)`

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            The output scores with shape (2 * num_nodes,), (num_nodes,)
        """
        score_pos = self.affine(h_pos, h_x).squeeze()
        score_neg = self.affine(h_neg, h_x).squeeze()
        if bias_pos is not None:
            score_pos = score_pos + bias_pos
        if bias_neg is not None:
            score_neg = score_neg + bias_neg

        logits = torch.cat((score_pos, score_neg), 0)

        return logits, score_pos


class DenseLayer(torch.nn.Module):
    """
    Description
    -----------
    Dense layer with a linear layer and an activation function
    """

    def __init__(
        self, in_dim: int, out_dim: int, act: str = "prelu", bias=True
    ):
        super(DenseLayer, self).__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.act_type = act.lower()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            torch.nn.init.zeros_(self.lin.bias)
        if self.act_type == "prelu":
            self.act = torch.nn.PReLU()
        else:
            self.act = torch.relu

    def forward(self, x):
        x = self.lin(x)
        return self.act(x)


class IndexSelect(torch.nn.Module):
    """
    Description
    -----------
    The index selection layer used by VIPool

    Parameters
    ----------
    pool_ratio : float
        The pooling ratio (for keeping nodes). For example,
        if `pool_ratio=0.8`, 80\% nodes will be preserved.
    hidden_dim : int
        The number of channels in node features.
    act : str, optional
        The activation function type.
        Default: :obj:`'prelu'`
    dist : int, optional
        DO NOT USE THIS PARAMETER
    """

    def __init__(
        self,
        pool_ratio: float,
        hidden_dim: int,
        act: str = "prelu",
        dist: int = 1,
    ):
        super(IndexSelect, self).__init__()
        self.pool_ratio = pool_ratio
        self.dist = dist
        self.dense = DenseLayer(hidden_dim, hidden_dim, act)
        self.discriminator = Discriminator(hidden_dim)
        self.gcn = GraphConvWithDropout(hidden_dim, hidden_dim)

    def forward(
        self,
        graph: DGLGraph,
        h_pos: Tensor,
        h_neg: Tensor,
        bias_pos: Optional[Tensor] = None,
        bias_neg: Optional[Tensor] = None,
    ):
        """
        Description
        -----------
        Perform index selection

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.
        h_pos : torch.Tensor
            The node features of positive samples
            It has the same shape as :obj:`h_x`
        h_neg : torch.Tensor
            The node features of negative samples
            It has the same shape as :obj:`h_x`
        bias_pos : torch.Tensor
            Bias parameter vector for positive scores
            shape: :obj:`(num_nodes)`
        bias_neg : torch.Tensor
            Bias parameter vector for negative scores
            shape: :obj:`(num_nodes)`
        """
        # compute scores
        h_pos = self.dense(h_pos)
        h_neg = self.dense(h_neg)
        embed = self.gcn(graph, h_pos)
        h_center = torch.sigmoid(embed)

        logit, logit_pos = self.discriminator(
            h_center, h_pos, h_neg, bias_pos, bias_neg
        )
        scores = torch.sigmoid(logit_pos)

        # sort scores
        scores, idx = torch.sort(scores, descending=True)

        # select top-k
        num_nodes = graph.num_nodes()
        num_select_nodes = int(self.pool_ratio * num_nodes)
        size_list = [num_select_nodes, num_nodes - num_select_nodes]
        select_scores, _ = torch.split(scores, size_list, dim=0)
        select_idx, non_select_idx = torch.split(idx, size_list, dim=0)

        return logit, select_scores, select_idx, non_select_idx, embed


class GraphPool(torch.nn.Module):
    """
    Description
    -----------
    The pooling module for graph

    Parameters
    ----------
    hidden_dim : int
        The number of channels of node features.
    use_gcn : bool, optional
        Whether use gcn in down sampling process.
        default: :obj:`False`
    """

    def __init__(self, hidden_dim: int, use_gcn=False):
        super(GraphPool, self).__init__()
        self.use_gcn = use_gcn
        self.down_sample_gcn = (
            GraphConvWithDropout(hidden_dim, hidden_dim) if use_gcn else None
        )

    def forward(
        self,
        graph: DGLGraph,
        feat: Tensor,
        select_idx: Tensor,
        non_select_idx: Optional[Tensor] = None,
        scores: Optional[Tensor] = None,
        pool_graph=False,
    ):
        """
        Description
        -----------
        Perform graph pooling.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The input graph
        feat : torch.Tensor
            The input node feature
        select_idx : torch.Tensor
            The index in fine graph of node from
            coarse graph, this is obtained from
            previous graph pooling layers.
        non_select_idx : torch.Tensor, optional
            The index that not included in output graph.
            default: :obj:`None`
        scores : torch.Tensor, optional
            Scores for nodes used for pooling and scaling.
            default: :obj:`None`
        pool_graph : bool, optional
            Whether perform graph pooling on graph topology.
            default: :obj:`False`
        """
        if self.use_gcn:
            feat = self.down_sample_gcn(graph, feat)

        feat = feat[select_idx]
        if scores is not None:
            feat = feat * scores.unsqueeze(-1)

        if pool_graph:
            num_node_batch = graph.batch_num_nodes()
            graph = dgl.node_subgraph(graph, select_idx)
            graph.set_batch_num_nodes(num_node_batch)
            return feat, graph
        else:
            return feat


class GraphUnpool(torch.nn.Module):
    """
    Description
    -----------
    The unpooling module for graph

    Parameters
    ----------
    hidden_dim : int
        The number of channels of node features.
    """

    def __init__(self, hidden_dim: int):
        super(GraphUnpool, self).__init__()
        self.up_sample_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)

    def forward(self, graph: DGLGraph, feat: Tensor, select_idx: Tensor):
        """
        Description
        -----------
        Perform graph unpooling

        Parameters
        ----------
        graph : dgl.DGLGraph
            The input graph
        feat : torch.Tensor
            The input node feature
        select_idx : torch.Tensor
            The index in fine graph of node from
            coarse graph, this is obtained from
            previous graph pooling layers.
        """
        fine_feat = torch.zeros(
            (graph.num_nodes(), feat.size(-1)), device=feat.device
        )
        fine_feat[select_idx] = feat
        fine_feat = self.up_sample_gcn(graph, fine_feat)
        return fine_feat
