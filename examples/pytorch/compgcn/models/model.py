#-*- coding:utf-8 -*-

import sys
sys.path.append("..")

import torch as th
import torch.nn as nn
import dgl.function as dglfn
import dgl.nn as dglnn
from dgl import DGLError

from .utils import ccorr
from data_utils import build_dummy_comp_data


class CompGraphConv(nn.Module):
    """
    One layer of composition graph convolutional network, including the 4 major computation forumual:

    ::math::

        1. Formula 1: node feature update
        $$h_v^{k+1}=f(\displaystyle\sum_{(u, r)\in \mathcal{N}(v)}W_{\lambda(r)}^k\phi(h_u^k, h_r^k))$$

        The $W_{\lambda(r)}$ has 3 types, $W_I, W_O, W_S$. Here, the aggregation is performed across
        relations.

        2. Formula 2: edge feature update
        $$h_r^{k+1}=W_{\text{rel}}^k h_r^k$$

        3. Formula 3: layer 0 of edge feature
        $$h_r^0 = W_{\text{rel}}^0z_r$$

        4. Formula 4: input layer of edge
        $$z_r=\displaystyle\sum_{b=1}^{\mathcal{B}}\alpha_{br}v_b$$

     """

    def __init__(self,
                 hid_dim,
                 rev_indicator,
                 comp_fn='sub',
                 activation=None,
                 bias=True,
                 dropout=0.0,
                 batchnorm=False):
        super(CompGraphConv, self).__init__()
        self.hid_dim = hid_dim
        self.rev_indicator = rev_indicator
        self.comp_fn = comp_fn
        self.activation = activation
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = batchnorm

        # define weights of 3 node matrics
        self.W_I = nn.Linear(self.hid_dim, self.hid_dim, bias=self.bias)
        self.W_O = nn.Linear(self.hid_dim, self.hid_dim, bias=self.bias)
        self.W_S = nn.Linear(self.hid_dim, self.hid_dim, bias=self.bias)

        # define weights of >0 edge matric
        self.W_rel = nn.Linear(self.hid_dim, self.hid_dim, bias=self.bias)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.hid_dim)

    def forward(self, g, n_in_feats, e_in_feats):
        """
        Input graph g should include reversed edges, and at the end of
        forward add self loop to each node

        Notes
        -----
        1. Node feature should have the same dimensions as edges so that
        composition function could apply to them.
        2. In CompGraphConv, the module assume the input graph has been
        added reversed edge for each original edge. For example, if there
        is a ('u' 'e' 'v') edge in the original graph, there must be a
        ('v' 'e_inv' 'u') edge, where e_inv is mannual added to the original
        graph AND edges in this type must have _inv in the end of edge name.

        Parameters
        ----------
        g: DGL Graph
            A DGL heterograph.

        n_in_feats: dict[str, Tensor]
            A dictionary of node type and node features.

        e_in_feats: dict[str, Tensor]
            A dictionary of edge type and edge features.

        """
        with g.local_scope():

            # handle each edge type relation sub_graph
            n_outputs = {ntype: [] for ntype in g.ntypes}

            for stype, etype, dtype in g.canonical_etypes:

                if etype not in e_in_feats:
                    continue

                # extract a subgraph in one edge type
                rel_graph = g[stype, etype, dtype]

                # check edge numbers
                if rel_graph.number_of_edges() == 0:
                    continue

                # assign node and edge features to the subgraph in the name of 'h'
                if stype not in n_in_feats or dtype not in n_in_feats:
                    continue
                else:
                    rel_graph.nodes[stype].data['h'] = n_in_feats[stype]
                    rel_graph.nodes[dtype].data['h'] = n_in_feats[dtype]

                    rel_graph.edges[etype].data['h'] = th.stack([e_in_feats[etype]] * rel_graph.number_of_edges())

                # compute norm values
                in_norm = rel_graph.in_degrees().float().clamp(min=1)
                in_norm = th.pow(in_norm, -0.5)

                # compute composition function
                if self.comp_fn == 'sub':
                    rel_graph.update_all(dglfn.u_sub_e(lhs_field='h', rhs_field='h', out='m'),
                                         dglfn.sum(msg='m', out='comp_h'))
                elif self.comp_fn == 'mul':
                    rel_graph.update_all(dglfn.u_mul_e(lhs_field='h', rhs_field='h', out='m'),
                                         dglfn.sum(msg='m', out='comp_h'))
                elif self.comp_fn == 'ccorr':
                    rel_graph.update_all(lambda edges: {'corr_h': ccorr(edges.src['h'], edges.data['h'])},
                                     dglfn.sum(msg='corr_h', out='comp_h'))
                else:
                    raise DGLError('Only supports sub, mul, and ccorr')

                # normalize composition results
                rel_graph.dstdata['comp_h'] = rel_graph.dstdata['comp_h'] * in_norm.view(-1, 1)

                # linear transformation by different weights and add bias values
                if str.endswith(etype, self.rev_indicator) or str.startswith(etype, self.rev_indicator):
                    rel_graph.dstdata['comp_h'] = self.W_O(rel_graph.dstdata['comp_h'])
                else:
                    rel_graph.dstdata['comp_h'] = self.W_I(rel_graph.dstdata['comp_h'])

                # add composition results to each node's list
                n_outputs[dtype].append(rel_graph.dstdata['comp_h'])

            # self loop
            self_rsts = {}
            for ntype in g.ntypes:
                self_rsts[ntype] = self.W_S(g.nodes[ntype].data['h'])

        # aggregate all relations' composition results
        n_out_feats = {}
        for ntype, alist in n_outputs.items():

            if len(alist) != 0:
                stacked = th.stack(alist, dim=0)
                n_out_feats[ntype] = th.sum(stacked, dim=0)

                # add self loop results and dropout
                n_out_feats[ntype] = self.dropout(n_out_feats[ntype]) * 2/3 + \
                                     self.dropout(self_rsts.get(ntype)) * 1/3

                # add batchnorm
                if self.batchnorm:
                    n_out_feats[ntype] = self.bn(n_out_feats[ntype])

                # use activate to compute non-linear
                if self.activation is not None:
                    n_out_feats[ntype] = self.activation(n_out_feats[ntype])

        # compute edge embedding of the next layer
        e_out_feats = {}
        for etype, feat in e_in_feats.items():
            e_out_feats[etype] = self.W_rel(e_in_feats[etype])

            # use activate to compute non-linear
            if self.activation is not None:
                e_out_feats[etype] = self.activation(e_out_feats[etype])

        return n_out_feats, e_out_feats


class CompGCN(nn.Module):
    """
    1. Will use dgl.nn.WeightBasis module to create basis vector of edge；

    """
    def __init__(self,
                 in_feat_dict,
                 hid_dim,
                 num_layers,
                 out_feat,
                 num_basis,
                 num_rel,
                 rev_indicator='_inv',
                 comp_fn='sub',
                 dropout=0.0,
                 activation=None,
                 batchnorm=False
                 ):
        super(CompGCN, self).__init__()
        self.in_feat_dict = in_feat_dict
        self.rel_emb_dim = hid_dim
        self.num_layer = num_layers
        self.out_feat = out_feat
        self.num_basis = num_basis
        self.num_rel = num_rel
        self.rev_indicator = rev_indicator

        self.comp_fn = comp_fn
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm

        # Input layer, complete two tasks
        # 1. define basis layer
        self.basis = dglnn.WeightBasis([self.rel_emb_dim], self.num_basis, self.num_rel)

        # 2. define matrices for convert node dimensions to relation embedding dimensions
        self.input_layer = nn.ModuleDict()
        for ntype, in_feat in in_feat_dict.items():
            self.input_layer[ntype] = nn.Linear(in_feat, self.rel_emb_dim, bias=True)

        # Hidden layers with n - 1 CompGraphConv layers
        self.layers = nn.ModuleList()
        for l in range(1, (self.num_layer - 1)):
            self.layers.append(CompGraphConv(self.rel_emb_dim,
                                             self.rev_indicator,
                                             comp_fn=self.comp_fn,
                                             activation=self.activation,
                                             bias=True,
                                             dropout=self.dropout,
                                             batchnorm=self.batchnorm))

        # Output layer with the output class
        self.output_layer = nn.Linear(self.rel_emb_dim, self.out_feat, bias=False)

    def forward(self, graph, nfeats):

        # Compute relation input
        h_e = {}
        basis_vec = self.basis.forward()
        for i, etype in enumerate(graph.etypes):
            h_e[etype] = basis_vec[i]

        # Convert node input dimension to relation dimension
        h_n = {}
        for ntype, feat in nfeats.items():
            h_n[ntype] = self.input_layer[ntype](feat)

        # Forward of n layers of CompGraphConv
        for layer in self.layers:
            h_n, h_e = layer(graph, h_n, h_e)

        # Forward of output layer
        outputs = {}
        for ntype, h in h_n.items():
            outputs[ntype] = self.output_layer(h)

        return outputs


if __name__ == '__main__':
    # Test with the original paper's diagram

    graph, n_feats, e_feats = build_dummy_comp_data()

    # Test for one layer of CompGraphConv module
    compgcn = CompGraphConv(hid_dim=5,
                            comp_fn='mul',
                            activation=nn.functional.relu,
                            bias=True,
                            dropout=0.1,
                            batchnorm=False)
    nfeats, efeats = compgcn(graph, n_feats, e_feats)
    for k, v in nfeats.items():
        print(k)
        print(v)
    for k, v in efeats.items():
        print(k)
        print(v)

    # Test for one CompGCN module
    in_feat_dict = {
        'user': 5,
        'city': 5,
        'country': 5,
        'film': 5
    }
    compgcn_model = CompGCN(in_feat_dict=in_feat_dict,
                            hid_dim=3,
                            num_layers=4,
                            out_feat=2,
                            num_basis=2,
                            num_rel=6,
                            comp_fn='sub',
                            dropout=0.0,
                            activation=None,
                            batchnorm=False
                            )
    logits = compgcn_model.forward(graph, n_feats)
    print(logits)