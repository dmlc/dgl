import torch
from dgl.nn import AvgPooling, MaxPooling
import torch.nn.functional as F
import torch.nn
from layers import HGPSLPool, WeightedGraphConv, ConvPoolReadout


class SimpleModel(torch.nn.Module):
    """Simplified version of Model. It has fixed number of layers (3)."""
    def __init__(self, in_feat:int, out_feat:int, hid_feat:int,
                 dropout:float=0.0, pool_ratio:float=0.5,
                 sample:bool=True, sparse:bool=True, sl:bool=True,
                 lamb:float=1.):
        super(SimpleModel, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hid_feat = hid_feat
        self.dropout = dropout
        self.pool_ratio = pool_ratio
        
        self.conv1 = WeightedGraphConv(in_feat, hid_feat)
        self.conv2 = WeightedGraphConv(hid_feat, hid_feat)
        self.conv3 = WeightedGraphConv(hid_feat, hid_feat)

        self.pool1 = HGPSLPool(hid_feat, ratio=pool_ratio, 
                               sample=sample, sparse=sparse,
                               sl=sl, lamb=lamb)
        self.pool2 = HGPSLPool(hid_feat, ratio=pool_ratio, 
                               sample=sample, sparse=sparse,
                               sl=sl, lamb=lamb)

        self.lin1 = torch.nn.Linear(hid_feat * 2, hid_feat)
        self.lin2 = torch.nn.Linear(hid_feat, hid_feat // 2)
        self.lin3 = torch.nn.Linear(hid_feat // 2, self.out_feat)
 
        self.avg_pool = AvgPooling()
        self.max_pool = MaxPooling()

    def forward(self, graph, n_feat):
        final_readout = None

        n_feat = F.relu(self.conv1(graph, n_feat))
        graph, n_feat, e_feat, _ = self.pool1(graph, n_feat, None)
        final_readout = torch.cat([self.avg_pool(graph, n_feat), self.max_pool(graph, n_feat)], dim=-1)

        n_feat = F.relu(self.conv2(graph, n_feat, e_feat))
        graph, n_feat, e_feat, _ = self.pool2(graph, n_feat, e_feat)
        final_readout = final_readout + torch.cat([self.avg_pool(graph, n_feat), self.max_pool(graph, n_feat)], dim=-1)

        n_feat = F.relu(self.conv3(graph, n_feat, e_feat))
        final_readout = final_readout + torch.cat([self.avg_pool(graph, n_feat), self.max_pool(graph, n_feat)], dim=-1)

        n_feat = F.relu(self.lin1(final_readout))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = F.relu(self.lin2(n_feat))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return F.log_softmax(n_feat, dim=-1)


class Model(torch.nn.Module):
    r"""

    Description
    -----------
    The graph classification model using HGP-SL pooling.

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels.
    out_feat : int
        The number of output node feature's channels.
    hid_feat : int
        The number of hidden state's channels.
    dropout : float, optional
        The dropout rate. Default: 0
    pool_ratio : float, optional
        The pooling ratio for each pooling layer. Default: 0.5
    conv_layers : int, optional
        The number of graph convolution and pooling layers. Default: 3
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency. 
        Currently we only support full graph. Default: :obj:`False`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    """
    def __init__(self, in_feat:int, out_feat:int, hid_feat:int,
                 dropout:float=0., pool_ratio:float=.5, conv_layers:int=3,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1.):
        super(Model, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hid_feat = hid_feat
        self.dropout = dropout
        self.num_layers = conv_layers
        self.pool_ratio = pool_ratio

        convpools = []
        for i in range(conv_layers):
            c_in = in_feat if i == 0 else hid_feat
            c_out = hid_feat
            use_pool = (i != conv_layers - 1)
            convpools.append(ConvPoolReadout(c_in, c_out, pool_ratio=pool_ratio,
                                             sample=sample, sparse=sparse, sl=sl,
                                             lamb=lamb, pool=use_pool))
        self.convpool_layers = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_feat * 2, hid_feat)
        self.lin2 = torch.nn.Linear(hid_feat, hid_feat // 2)
        self.lin3 = torch.nn.Linear(hid_feat // 2, self.out_feat)

    def forward(self, graph, n_feat):
        final_readout = None
        e_feat = None

        for i in range(self.num_layers):
            graph, n_feat, e_feat, readout = self.convpool_layers[i](graph, n_feat, e_feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        
        n_feat = F.relu(self.lin1(final_readout))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = F.relu(self.lin2(n_feat))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return F.log_softmax(n_feat, dim=-1)
