from typing import List, Tuple, Union

from layers import *
import dgl.function as fn
import torch
import torch.nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SortPooling


class GraphCrossModule(torch.nn.Module):
    """
    Description
    -----------
    The Graph Cross Module used by Graph Cross Networks.
    This module only contains graph cross layers.

    Parameters
    ----------
    pool_ratios : Union[float, List[float]]
        The pooling ratios (for keeping nodes) for each layer.
        For example, if `pool_ratio=0.8`, 80\% nodes will be preserved.
        If a single float number is given, all pooling layers will have the
        same pooling ratio.
    in_dim : int
        The number of input node feature channels.
    out_dim : int
        The number of output node feature channels.
    hidden_dim : int
        The number of hidden node feature channels.
    cross_weight : float, optional
        The weight parameter used in graph cross layers
        Default: :obj:`1.0`
    fuse_weight : float, optional
        The weight parameter used at the end of GXN for channel fusion.
        Default: :obj:`1.0`
    """

    def __init__(
        self,
        pool_ratios: Union[float, List[float]],
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        cross_weight: float = 1.0,
        fuse_weight: float = 1.0,
        dist: int = 1,
        num_cross_layers: int = 2,
    ):
        super(GraphCrossModule, self).__init__()
        if isinstance(pool_ratios, float):
            pool_ratios = (pool_ratios, pool_ratios)
        self.cross_weight = cross_weight
        self.fuse_weight = fuse_weight
        self.num_cross_layers = num_cross_layers

        # build network
        self.start_gcn_scale1 = GraphConvWithDropout(in_dim, hidden_dim)
        self.start_gcn_scale2 = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.end_gcn = GraphConvWithDropout(2 * hidden_dim, out_dim)

        self.index_select_scale1 = IndexSelect(
            pool_ratios[0], hidden_dim, act="prelu", dist=dist
        )
        self.index_select_scale2 = IndexSelect(
            pool_ratios[1], hidden_dim, act="prelu", dist=dist
        )
        self.start_pool_s12 = GraphPool(hidden_dim)
        self.start_pool_s23 = GraphPool(hidden_dim)
        self.end_unpool_s21 = GraphUnpool(hidden_dim)
        self.end_unpool_s32 = GraphUnpool(hidden_dim)

        self.s1_l1_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.s1_l2_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.s1_l3_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)

        self.s2_l1_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.s2_l2_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.s2_l3_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)

        self.s3_l1_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.s3_l2_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)
        self.s3_l3_gcn = GraphConvWithDropout(hidden_dim, hidden_dim)

        if num_cross_layers >= 1:
            self.pool_s12_1 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s21_1 = GraphUnpool(hidden_dim)
            self.pool_s23_1 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s32_1 = GraphUnpool(hidden_dim)
        if num_cross_layers >= 2:
            self.pool_s12_2 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s21_2 = GraphUnpool(hidden_dim)
            self.pool_s23_2 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s32_2 = GraphUnpool(hidden_dim)

    def forward(self, graph, feat):
        # start of scale-1
        graph_scale1 = graph
        feat_scale1 = self.start_gcn_scale1(graph_scale1, feat)
        feat_origin = feat_scale1
        feat_scale1_neg = feat_scale1[
            torch.randperm(feat_scale1.size(0))
        ]  # negative samples
        (
            logit_s1,
            scores_s1,
            select_idx_s1,
            non_select_idx_s1,
            feat_down_s1,
        ) = self.index_select_scale1(graph_scale1, feat_scale1, feat_scale1_neg)
        feat_scale2, graph_scale2 = self.start_pool_s12(
            graph_scale1,
            feat_scale1,
            select_idx_s1,
            non_select_idx_s1,
            scores_s1,
            pool_graph=True,
        )

        # start of scale-2
        feat_scale2 = self.start_gcn_scale2(graph_scale2, feat_scale2)
        feat_scale2_neg = feat_scale2[
            torch.randperm(feat_scale2.size(0))
        ]  # negative samples
        (
            logit_s2,
            scores_s2,
            select_idx_s2,
            non_select_idx_s2,
            feat_down_s2,
        ) = self.index_select_scale2(graph_scale2, feat_scale2, feat_scale2_neg)
        feat_scale3, graph_scale3 = self.start_pool_s23(
            graph_scale2,
            feat_scale2,
            select_idx_s2,
            non_select_idx_s2,
            scores_s2,
            pool_graph=True,
        )

        # layer-1
        res_s1_0, res_s2_0, res_s3_0 = feat_scale1, feat_scale2, feat_scale3

        feat_scale1 = F.relu(self.s1_l1_gcn(graph_scale1, feat_scale1))
        feat_scale2 = F.relu(self.s2_l1_gcn(graph_scale2, feat_scale2))
        feat_scale3 = F.relu(self.s3_l1_gcn(graph_scale3, feat_scale3))

        if self.num_cross_layers >= 1:
            feat_s12_fu = self.pool_s12_1(
                graph_scale1,
                feat_scale1,
                select_idx_s1,
                non_select_idx_s1,
                scores_s1,
            )
            feat_s21_fu = self.unpool_s21_1(
                graph_scale1, feat_scale2, select_idx_s1
            )
            feat_s23_fu = self.pool_s23_1(
                graph_scale2,
                feat_scale2,
                select_idx_s2,
                non_select_idx_s2,
                scores_s2,
            )
            feat_s32_fu = self.unpool_s32_1(
                graph_scale2, feat_scale3, select_idx_s2
            )

            feat_scale1 = (
                feat_scale1 + self.cross_weight * feat_s21_fu + res_s1_0
            )
            feat_scale2 = (
                feat_scale2
                + self.cross_weight * (feat_s12_fu + feat_s32_fu) / 2
                + res_s2_0
            )
            feat_scale3 = (
                feat_scale3 + self.cross_weight * feat_s23_fu + res_s3_0
            )

        # layer-2
        feat_scale1 = F.relu(self.s1_l2_gcn(graph_scale1, feat_scale1))
        feat_scale2 = F.relu(self.s2_l2_gcn(graph_scale2, feat_scale2))
        feat_scale3 = F.relu(self.s3_l2_gcn(graph_scale3, feat_scale3))

        if self.num_cross_layers >= 2:
            feat_s12_fu = self.pool_s12_2(
                graph_scale1,
                feat_scale1,
                select_idx_s1,
                non_select_idx_s1,
                scores_s1,
            )
            feat_s21_fu = self.unpool_s21_2(
                graph_scale1, feat_scale2, select_idx_s1
            )
            feat_s23_fu = self.pool_s23_2(
                graph_scale2,
                feat_scale2,
                select_idx_s2,
                non_select_idx_s2,
                scores_s2,
            )
            feat_s32_fu = self.unpool_s32_2(
                graph_scale2, feat_scale3, select_idx_s2
            )

            cross_weight = self.cross_weight * 0.05
            feat_scale1 = feat_scale1 + cross_weight * feat_s21_fu
            feat_scale2 = (
                feat_scale2 + cross_weight * (feat_s12_fu + feat_s32_fu) / 2
            )
            feat_scale3 = feat_scale3 + cross_weight * feat_s23_fu

        # layer-3
        feat_scale1 = F.relu(self.s1_l3_gcn(graph_scale1, feat_scale1))
        feat_scale2 = F.relu(self.s2_l3_gcn(graph_scale2, feat_scale2))
        feat_scale3 = F.relu(self.s3_l3_gcn(graph_scale3, feat_scale3))

        # final layers
        feat_s3_out = (
            self.end_unpool_s32(graph_scale2, feat_scale3, select_idx_s2)
            + feat_down_s2
        )
        feat_s2_out = self.end_unpool_s21(
            graph_scale1, feat_scale2 + feat_s3_out, select_idx_s1
        )
        feat_agg = (
            feat_scale1
            + self.fuse_weight * feat_s2_out
            + self.fuse_weight * feat_down_s1
        )
        feat_agg = torch.cat((feat_agg, feat_origin), dim=1)
        feat_agg = self.end_gcn(graph_scale1, feat_agg)

        return feat_agg, logit_s1, logit_s2


class GraphCrossNet(torch.nn.Module):
    """
    Description
    -----------
    The Graph Cross Network.

    Parameters
    ----------
    in_dim : int
        The number of input node feature channels.
    out_dim : int
        The number of output node feature channels.
    edge_feat_dim : int, optional
        The number of input edge feature channels. Edge feature
        will be passed to a Linear layer and concatenated to
        input node features. Default: :obj:`0`
    hidden_dim : int, optional
        The number of hidden node feature channels.
        Default: :obj:`96`
    pool_ratios : Union[float, List[float]], optional
        The pooling ratios (for keeping nodes) for each layer.
        For example, if `pool_ratio=0.8`, 80\% nodes will be preserved.
        If a single float number is given, all pooling layers will have the
        same pooling ratio.
        Default: :obj:`[0.9, 0.7]`
    readout_nodes : int, optional
        Number of nodes perserved in the final sort pool operation.
        Default: :obj:`30`
    conv1d_dims : List[int], optional
        The number of kernels of Conv1d operations.
        Default: :obj:`[16, 32]`
    conv1d_kws : List[int], optional
        The kernel size of Conv1d.
        Default: :obj:`[5]`
    cross_weight : float, optional
        The weight parameter used in graph cross layers
        Default: :obj:`1.0`
    fuse_weight : float, optional
        The weight parameter used at the end of GXN for channel fusion.
        Default: :obj:`1.0`
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_feat_dim: int = 0,
        hidden_dim: int = 96,
        pool_ratios: Union[List[float], float] = [0.9, 0.7],
        readout_nodes: int = 30,
        conv1d_dims: List[int] = [16, 32],
        conv1d_kws: List[int] = [5],
        cross_weight: float = 1.0,
        fuse_weight: float = 1.0,
        dist: int = 1,
    ):
        super(GraphCrossNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.readout_nodes = readout_nodes
        conv1d_kws = [hidden_dim] + conv1d_kws

        if edge_feat_dim > 0:
            self.in_dim += hidden_dim
            self.e2l_lin = torch.nn.Linear(edge_feat_dim, hidden_dim)
        else:
            self.e2l_lin = None

        self.gxn = GraphCrossModule(
            pool_ratios,
            in_dim=self.in_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            cross_weight=cross_weight,
            fuse_weight=fuse_weight,
            dist=dist,
        )
        self.sortpool = SortPooling(readout_nodes)

        # final updates
        self.final_conv1 = torch.nn.Conv1d(
            1, conv1d_dims[0], kernel_size=conv1d_kws[0], stride=conv1d_kws[0]
        )
        self.final_maxpool = torch.nn.MaxPool1d(2, 2)
        self.final_conv2 = torch.nn.Conv1d(
            conv1d_dims[0], conv1d_dims[1], kernel_size=conv1d_kws[1], stride=1
        )
        self.final_dense_dim = int((readout_nodes - 2) / 2 + 1)
        self.final_dense_dim = (
            self.final_dense_dim - conv1d_kws[1] + 1
        ) * conv1d_dims[1]

        if self.out_dim > 0:
            self.out_lin = torch.nn.Linear(self.final_dense_dim, out_dim)

        self.init_weights()

    def init_weights(self):
        if self.e2l_lin is not None:
            torch.nn.init.xavier_normal_(self.e2l_lin.weight)
        torch.nn.init.xavier_normal_(self.final_conv1.weight)
        torch.nn.init.xavier_normal_(self.final_conv2.weight)
        if self.out_dim > 0:
            torch.nn.init.xavier_normal_(self.out_lin.weight)

    def forward(
        self,
        graph: DGLGraph,
        node_feat: Tensor,
        edge_feat: Optional[Tensor] = None,
    ):
        num_batch = graph.batch_size
        if edge_feat is not None:
            edge_feat = self.e2l_lin(edge_feat)
            with graph.local_scope():
                graph.edata["he"] = edge_feat
                graph.update_all(fn.copy_e("he", "m"), fn.sum("m", "hn"))
                edge2node_feat = graph.ndata.pop("hn")
                node_feat = torch.cat((node_feat, edge2node_feat), dim=1)

        node_feat, logits1, logits2 = self.gxn(graph, node_feat)
        batch_sortpool_feats = self.sortpool(graph, node_feat)

        # final updates
        to_conv1d = batch_sortpool_feats.unsqueeze(1)
        conv1d_result = F.relu(self.final_conv1(to_conv1d))
        conv1d_result = self.final_maxpool(conv1d_result)
        conv1d_result = F.relu(self.final_conv2(conv1d_result))

        to_dense = conv1d_result.view(num_batch, -1)
        if self.out_dim > 0:
            out = F.relu(self.out_lin(to_dense))
        else:
            out = to_dense

        return out, logits1, logits2


class GraphClassifier(torch.nn.Module):
    """
    Description
    -----------
    Graph Classifier for graph classification.
    GXN + MLP
    """

    def __init__(self, args):
        super(GraphClassifier, self).__init__()
        self.gxn = GraphCrossNet(
            in_dim=args.in_dim,
            out_dim=args.embed_dim,
            edge_feat_dim=args.edge_feat_dim,
            hidden_dim=args.hidden_dim,
            pool_ratios=args.pool_ratios,
            readout_nodes=args.readout_nodes,
            conv1d_dims=args.conv1d_dims,
            conv1d_kws=args.conv1d_kws,
            cross_weight=args.cross_weight,
            fuse_weight=args.fuse_weight,
        )
        self.lin1 = torch.nn.Linear(args.embed_dim, args.final_dense_hidden_dim)
        self.lin2 = torch.nn.Linear(args.final_dense_hidden_dim, args.out_dim)
        self.dropout = args.dropout

    def forward(
        self,
        graph: DGLGraph,
        node_feat: Tensor,
        edge_feat: Optional[Tensor] = None,
    ):
        embed, logits1, logits2 = self.gxn(graph, node_feat, edge_feat)
        logits = F.relu(self.lin1(embed))
        if self.dropout > 0:
            logits = F.dropout(logits, p=self.dropout, training=self.training)
        logits = self.lin2(logits)
        return F.log_softmax(logits, dim=1), logits1, logits2
