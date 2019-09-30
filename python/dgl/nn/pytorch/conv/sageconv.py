"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn
from torch.nn import functional as F

from .... import function as fn


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_feats)),
             m.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat
        if self._aggre_type == 'mean':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().float()
            degs = degs.to(feat.device)
            h_neigh = (graph.ndata['neigh'] + graph.ndata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = F.relu(self.fc_pool(feat))
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'lstm':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.ndata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
