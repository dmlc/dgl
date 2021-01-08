"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.sampling import select_topk
from functools import partial
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F

class HardGAO(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 k=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=F.elu,
                 allow_zero_in_degree=False):
        super(HardGAO, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.k = k
        # Initialize Parameters for Additive Attention
        print('out feats:',self.out_feats)
        self.fc = nn.Linear(
            self.in_feats, self.out_feats * self.num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        # Initialize Parameters for Hard Projection
        self.p = nn.Parameter(torch.FloatTensor(size=(1,in_feats)))
        # Initialize Dropouts
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        # TODO: Maybe need exactly same initialization
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.p,gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def n2e_weight_transfer(self,edges):
        y = edges.src['y']
        return {'y':y}

    def forward(self, graph, feat, get_attention=False):
            # projection process to get importance vector y
            graph.ndata['y'] = torch.abs(torch.matmul(self.p,feat.T).view(-1))/torch.norm(self.p,p=2)
            # Use edge message passing function to get the weight from src node
            graph.apply_edges(self.n2e_weight_transfer)
            # Select Top k neighbors
            subgraph = select_topk(graph,self.k,'y')
            # Sigmoid as information threshold
            subgraph.ndata['y'] = torch.sigmoid(subgraph.ndata['y'])*2
            # Using vector matrix elementwise mul for acceleration
            feat = subgraph.ndata['y'].view(-1,1)*feat
            feat = self.feat_drop(feat)
            feat = self.fc(feat).view(-1, self.num_heads, self.out_feats)
            el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # Assign the value on the subgraph
            subgraph.srcdata.update({'ft': feat, 'el': el})
            subgraph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            subgraph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(subgraph.edata.pop('e'))
            # compute softmax
            subgraph.edata['a'] = self.attn_drop(edge_softmax(subgraph, e))
            # message passing
            subgraph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = subgraph.dstdata['ft']
            
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, subgraph.edata['a']
            else:
                return rst

class GAM(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 head=1,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=True,
                 activation=F.elu,
                 k=8,
                 model='hgat'
                 ):
        super(GAM,self).__init__()
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.residual = residual
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.head = head
        self.k = k
        self.negative_slope = negative_slope
        if model == 'hgat':
            self.hgao = HardGAO(in_dim,out_dim,
                                self.head,
                                k=self.k,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout,
                                negative_slope=self.negative_slope,
                                activation=activation
            )
        elif model == 'gat':
            self.hgao = GATConv(in_dim,out_dim,
                                self.head,
                                feat_drop=feat_drop,
                                attn_drop=attn_drop,
                                negative_slope=negative_slope,
                                activation=activation)
        else:
            raise NotImplementedError("No other model supported please use `hgat` or `gat`")
        # Residual module
        if self.residual:
            if self.in_dim==self.out_dim:
                self.res_m = Identity()
            else:
                self.res_m = nn.Linear(self.in_dim,self.out_dim*head)
    
    def forward(self,g,n_feats):
        h = self.hgao(g,n_feats)
        if self.residual:
            ret = h + self.res_m(n_feats).view(h.shape[0],-1,h.shape[2])
        else:
            ret = h
        return ret

class HardGAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 model,k):
        super(HardGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        gat_layer = partial(GAM,k=k) if model == 'hgat' else GATConv
        muls = heads
        self.model = model
        # input projection (no residual)
        self.gat_layers.append(gat_layer(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(gat_layer(
                num_hidden*muls[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(gat_layer(
            num_hidden*muls[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
