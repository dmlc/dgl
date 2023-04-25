import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
from dgl.base import DGLError


class DiffConv(nn.Module):
    """DiffConv is the implementation of diffusion convolution from paper DCRNN
    It will compute multiple diffusion matrix and perform multiple diffusion conv on it,
    this layer can be used for traffic prediction, pedamic model.
    Parameter
    ==========
    in_feats : int
        number of input feature

    out_feats : int
        number of output feature

    k : int
        number of diffusion steps

    dir : str [both/in/out]
        direction of diffusion convolution
        From paper default both direction
    """

    def __init__(
        self, in_feats, out_feats, k, in_graph_list, out_graph_list, dir="both"
    ):
        super(DiffConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k
        self.dir = dir
        self.num_graphs = self.k - 1 if self.dir == "both" else 2 * self.k - 2
        self.project_fcs = nn.ModuleList()
        for i in range(self.num_graphs):
            self.project_fcs.append(
                nn.Linear(self.in_feats, self.out_feats, bias=False)
            )
        self.merger = nn.Parameter(torch.randn(self.num_graphs + 1))
        self.in_graph_list = in_graph_list
        self.out_graph_list = out_graph_list

    @staticmethod
    def attach_graph(g, k):
        device = g.device
        out_graph_list = []
        in_graph_list = []
        wadj, ind, outd = DiffConv.get_weight_matrix(g)
        adj = sparse.coo_matrix(wadj / outd.cpu().numpy())
        outg = dgl.from_scipy(adj, eweight_name="weight").to(device)
        outg.edata["weight"] = outg.edata["weight"].float().to(device)
        out_graph_list.append(outg)
        for i in range(k - 1):
            out_graph_list.append(
                DiffConv.diffuse(out_graph_list[-1], wadj, outd)
            )
        adj = sparse.coo_matrix(wadj.T / ind.cpu().numpy())
        ing = dgl.from_scipy(adj, eweight_name="weight").to(device)
        ing.edata["weight"] = ing.edata["weight"].float().to(device)
        in_graph_list.append(ing)
        for i in range(k - 1):
            in_graph_list.append(
                DiffConv.diffuse(in_graph_list[-1], wadj.T, ind)
            )
        return out_graph_list, in_graph_list

    @staticmethod
    def get_weight_matrix(g):
        adj = g.adj_external(scipy_fmt="coo")
        ind = g.in_degrees()
        outd = g.out_degrees()
        weight = g.edata["weight"]
        adj.data = weight.cpu().numpy()
        return adj, ind, outd

    @staticmethod
    def diffuse(progress_g, weighted_adj, degree):
        device = progress_g.device
        progress_adj = progress_g.adj_external(scipy_fmt="coo")
        progress_adj.data = progress_g.edata["weight"].cpu().numpy()
        ret_adj = sparse.coo_matrix(
            progress_adj @ (weighted_adj / degree.cpu().numpy())
        )
        ret_graph = dgl.from_scipy(ret_adj, eweight_name="weight").to(device)
        ret_graph.edata["weight"] = ret_graph.edata["weight"].float().to(device)
        return ret_graph

    def forward(self, g, x):
        feat_list = []
        if self.dir == "both":
            graph_list = self.in_graph_list + self.out_graph_list
        elif self.dir == "in":
            graph_list = self.in_graph_list
        elif self.dir == "out":
            graph_list = self.out_graph_list

        for i in range(self.num_graphs):
            g = graph_list[i]
            with g.local_scope():
                g.ndata["n"] = self.project_fcs[i](x)
                g.update_all(
                    fn.u_mul_e("n", "weight", "e"), fn.sum("e", "feat")
                )
                feat_list.append(g.ndata["feat"])
                # Each feat has shape [N,q_feats]
        feat_list.append(self.project_fcs[-1](x))
        feat_list = torch.cat(feat_list).view(
            len(feat_list), -1, self.out_feats
        )
        ret = (
            (self.merger * feat_list.permute(1, 2, 0)).permute(2, 0, 1).mean(0)
        )
        return ret
