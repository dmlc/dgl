import dgl
import dgl.function as fn
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import AvgPooling, GraphConv, MaxPooling
from dgl.ops import edge_softmax

from functions import edge_sparsemax
from torch import Tensor
from torch.nn import Parameter
from utils import get_batch_id, topk


class WeightedGraphConv(GraphConv):
    r"""
    Description
    -----------
    GraphConv with edge weights on homogeneous graphs.
    If edge weights are not given, directly call GraphConv instead.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    n_feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`
    """

    def forward(self, graph: DGLGraph, n_feat, e_feat=None):
        if e_feat is None:
            return super(WeightedGraphConv, self).forward(graph, n_feat)

        with graph.local_scope():
            if self.weight is not None:
                n_feat = torch.matmul(n_feat, self.weight)
            src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
            src_norm = src_norm.view(-1, 1)
            dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            dst_norm = dst_norm.view(-1, 1)
            n_feat = n_feat * src_norm
            graph.ndata["h"] = n_feat
            graph.edata["e"] = e_feat
            graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))
            n_feat = graph.ndata.pop("h")
            n_feat = n_feat * dst_norm
            if self.bias is not None:
                n_feat = n_feat + self.bias
            if self._activation is not None:
                n_feat = self._activation(n_feat)
            return n_feat


class NodeInfoScoreLayer(nn.Module):
    r"""
    Description
    -----------
    Compute a score for each node for sort-pooling. The score of each node
    is computed via the absolute difference of its first-order random walk
    result and its features.

    Arguments
    ---------
    sym_norm : bool, optional
        If true, use symmetric norm for adjacency.
        Default: :obj:`True`

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`

    Returns
    -------
    Tensor
        Score for each node.
    """

    def __init__(self, sym_norm: bool = True):
        super(NodeInfoScoreLayer, self).__init__()
        self.sym_norm = sym_norm

    def forward(self, graph: dgl.DGLGraph, feat: Tensor, e_feat: Tensor):
        with graph.local_scope():
            if self.sym_norm:
                src_norm = torch.pow(
                    graph.out_degrees().float().clamp(min=1), -0.5
                )
                src_norm = src_norm.view(-1, 1).to(feat.device)
                dst_norm = torch.pow(
                    graph.in_degrees().float().clamp(min=1), -0.5
                )
                dst_norm = dst_norm.view(-1, 1).to(feat.device)

                src_feat = feat * src_norm

                graph.ndata["h"] = src_feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))

                dst_feat = graph.ndata.pop("h") * dst_norm
                feat = feat - dst_feat
            else:
                dst_norm = 1.0 / graph.in_degrees().float().clamp(min=1)
                dst_norm = dst_norm.view(-1, 1)

                graph.ndata["h"] = feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))

                feat = feat - dst_norm * graph.ndata.pop("h")

            score = torch.sum(torch.abs(feat), dim=1)
            return score


class HGPSLPool(nn.Module):
    r"""

    Description
    -----------
    The HGP-SL pooling layer from
    `Hierarchical Graph Pooling with Structure Learning <https://arxiv.org/pdf/1911.05954.pdf>`

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels
    ratio : float, optional
        Pooling ratio. Default: 0.8
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency.
        Currently we only support full graph. Default: :obj:`False`
    sym_score_norm : bool, optional
        Use symmetric norm for adjacency or not. Default: :obj:`True`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    negative_slop : float, optional
        Negative slop for leaky_relu. Default: 0.2

    Returns
    -------
    DGLGraph
        The pooled graph.
    torch.Tensor
        Node features
    torch.Tensor
        Edge features
    torch.Tensor
        Permutation index
    """

    def __init__(
        self,
        in_feat: int,
        ratio=0.8,
        sample=True,
        sym_score_norm=True,
        sparse=True,
        sl=True,
        lamb=1.0,
        negative_slop=0.2,
        k_hop=3,
    ):
        super(HGPSLPool, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop

        self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att.data)

    def forward(self, graph: DGLGraph, feat: Tensor, e_feat=None):
        # top-k pool first
        if e_feat is None:
            e_feat = torch.ones(
                (graph.num_edges(),), dtype=feat.dtype, device=feat.device
            )
        batch_num_nodes = graph.batch_num_nodes()
        x_score = self.calc_info_score(graph, feat, e_feat)
        perm, next_batch_num_nodes = topk(
            x_score, self.ratio, get_batch_id(batch_num_nodes), batch_num_nodes
        )
        feat = feat[perm]
        pool_graph = None
        if not self.sample or not self.sl:
            # pool graph
            graph.edata["e"] = e_feat
            pool_graph = dgl.node_subgraph(graph, perm)
            e_feat = pool_graph.edata.pop("e")
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        # no structure learning layer, directly return.
        if not self.sl:
            return pool_graph, feat, e_feat, perm

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each
            # pair of nodes is time consuming. To accelerate this process,
            # we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.

            # first build multi-hop graph
            row, col = graph.all_edges()
            num_nodes = graph.num_nodes()

            scipy_adj = scipy.sparse.coo_matrix(
                (
                    e_feat.detach().cpu(),
                    (row.detach().cpu(), col.detach().cpu()),
                ),
                shape=(num_nodes, num_nodes),
            )
            for _ in range(self.k_hop):
                two_hop = scipy_adj**2
                two_hop = two_hop * (1e-5 / two_hop.max())
                scipy_adj = two_hop + scipy_adj
            row, col = scipy_adj.nonzero()
            row = torch.tensor(row, dtype=torch.long, device=graph.device)
            col = torch.tensor(col, dtype=torch.long, device=graph.device)
            e_feat = torch.tensor(
                scipy_adj.data, dtype=torch.float, device=feat.device
            )

            # perform pooling on multi-hop graph
            mask = perm.new_full((num_nodes,), -1)
            i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
            mask[perm] = i
            row, col = mask[row], mask[col]
            mask = (row >= 0) & (col >= 0)
            row, col = row[mask], col[mask]
            e_feat = e_feat[mask]

            # add remaining self loops
            mask = row != col
            num_nodes = perm.size(0)  # num nodes after pool
            loop_index = torch.arange(
                0, num_nodes, dtype=row.dtype, device=row.device
            )
            inv_mask = ~mask
            loop_weight = torch.full(
                (num_nodes,), 0, dtype=e_feat.dtype, device=e_feat.device
            )
            remaining_e_feat = e_feat[inv_mask]
            if remaining_e_feat.numel() > 0:
                loop_weight[row[inv_mask]] = remaining_e_feat
            e_feat = torch.cat([e_feat[mask], loop_weight], dim=0)
            row, col = row[mask], col[mask]
            row = torch.cat([row, loop_index], dim=0)
            col = torch.cat([col, loop_index], dim=0)

            # attention scores
            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(
                dim=-1
            )
            weights = (
                F.leaky_relu(weights, self.negative_slop) + e_feat * self.lamb
            )

            # sl and normalization
            sl_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(sl_graph, weights)
            else:
                weights = edge_softmax(sl_graph, weights)

            # get final graph
            mask = torch.abs(weights) > 0
            row, col, weights = row[mask], col[mask], weights[mask]
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            e_feat = weights

        else:
            # Learning the possible edge weights between each pair of
            # nodes in the pooled subgraph, relative slower.

            # construct complete graphs for all graph in the batch
            # use dense to build, then transform to sparse.
            # maybe there's more efficient way?
            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat(
                [
                    batch_num_nodes.new_zeros(1),
                    batch_num_nodes.cumsum(dim=0)[:-1],
                ],
                dim=0,
            )
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros(
                (pool_graph.num_nodes(), pool_graph.num_nodes()),
                dtype=torch.float,
                device=feat.device,
            )
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.0
            row, col = torch.nonzero(dense_adj).t().contiguous()

            # compute weights for node-pairs
            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(
                dim=-1
            )
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights

            # add pooled graph structure to weight matrix
            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]
            del dense_adj
            torch.cuda.empty_cache()

            # edge softmax/sparsemax
            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)

            # get new e_feat and graph structure, clean up.
            mask = torch.abs(weights) > 1e-9
            row, col, weights = row[mask], col[mask], weights[mask]
            e_feat = weights
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        return pool_graph, feat, e_feat, perm


class ConvPoolReadout(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        pool_ratio=0.8,
        sample: bool = False,
        sparse: bool = True,
        sl: bool = True,
        lamb: float = 1.0,
        pool: bool = True,
    ):
        super(ConvPoolReadout, self).__init__()
        self.use_pool = pool
        self.conv = WeightedGraphConv(in_feat, out_feat)
        if pool:
            self.pool = HGPSLPool(
                out_feat,
                ratio=pool_ratio,
                sparse=sparse,
                sample=sample,
                sl=sl,
                lamb=lamb,
            )
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):
        out = F.relu(self.conv(graph, feature, e_feat))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)
        readout = torch.cat(
            [self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1
        )
        return graph, out, e_feat, readout
