import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from ogb.graphproppred.mol_encoder import AtomEncoder
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from modules import norm_layer
from layers import GENConv


class DeeperGCN(nn.Module):
    r"""

    Description
    -----------
    Introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    dataset: str
        Name of ogb dataset.
    node_feat_dim: int
        Size of node feature dimension.
    edge_feat_dim: int
        Size of edge feature dimension.
    hid_dim: int
        Size of hidden dimension.
    out_dim: int
        Size of output dimension.
    num_layers: int
        Number of graph convolutional layers.
    dropout: float
        Dropout rate. Default is 0.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    pooling: str
        Type of ('sum', 'mean', 'max') pooling layer. Default is 'mean'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    lean_beta: bool
        Whether beta is a learnable weight. Default is False.
    aggr: str
        Type of aggregator scheme ('softmax', 'power'). Default is 'softmax'.
    mlp_layers: int
        Number of MLP layers in message normalization. Default is 1.
    """
    def __init__(self,
                 dataset,
                 node_feat_dim,
                 edge_feat_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout=0.,
                 norm='batch',
                 pooling='mean',
                 beta=1.0,
                 learn_beta=False,
                 aggr='softmax',
                 mlp_layers=1):
        super(DeeperGCN, self).__init__()
        
        self.dataset = dataset
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers):
            conv = GENConv(dataset=dataset,
                           in_dim=hid_dim,
                           out_dim=hid_dim,
                           aggregator=aggr,
                           beta=beta,
                           learn_beta=learn_beta,
                           mlp_layers=mlp_layers,
                           norm=norm)
            
            self.gcns.append(conv)
            self.norms.append(norm_layer(norm, hid_dim))

        if self.dataset == 'ogbg-molhiv':
            self.node_encoder = AtomEncoder(hid_dim)
        elif self.dataset == 'ogbg-ppa':
            self.node_encoder = nn.Linear(node_feat_dim, hid_dim)
            self.edge_encoder = nn.Linear(edge_feat_dim, hid_dim)
        else:
            raise ValueError(f'Dataset {dataset} is not supported.')

        if pooling == 'sum':
            self.pooling = SumPooling()
        elif pooling == 'mean':
            self.pooling = AvgPooling()
        elif pooling == 'max':
            self.pooling = MaxPooling()
        else:
            raise NotImplementedError(f'{pooling} is not supported.')
        
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, edge_feats, node_feats=None):
        with g.local_scope():
            if self.dataset == 'ogbg-molhiv':
                hv = self.node_encoder(node_feats)
                he = edge_feats
            else:
                # node features are initialized via a Sum aggr in ogbg-ppa
                g.edata['h'] = edge_feats
                g.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'h'))
                hv = self.node_encoder(g.ndata['h'])
                he = self.edge_encoder(edge_feats)

            for layer in range(self.num_layers):
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                hv = self.gcns[layer](g, hv1, he) + hv

            h_g = self.pooling(g, hv)

            return self.output(h_g)
