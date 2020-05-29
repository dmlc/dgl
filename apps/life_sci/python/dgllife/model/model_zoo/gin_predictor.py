"""GIN-based model for regression and classification on graphs."""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl
import torch.nn as nn

from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling

from ..gnn.gin import GIN

__all__ = ['GINPredictor']

# pylint: disable=W0221
class GINPredictor(nn.Module):
    """GIN-based model for regression and classification on graphs.

    GIN was first introduced in `How Powerful Are Graph Neural Networks
    <https://arxiv.org/abs/1810.00826>`__ for general graph property
    prediction problems. It was further extended in `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    for pre-training and semi-supervised learning on large-scale datasets.

    For classification tasks, the output will be logits, i.e. values before
    sigmoid or softmax.

    Parameters
    ----------
    num_node_emb_list : list of int
        num_node_emb_list[i] gives the number of items to embed for the
        i-th categorical node feature variables. E.g. num_node_emb_list[0] can be
        the number of atom types and num_node_emb_list[1] can be the number of
        atom chirality types.
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``'concat'``, ``'last'``, ``'max'`` and
        ``'sum'``. Default to 'last'.

        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5.
    readout : str
        Readout for computing graph representations out of node representations, which
        can be ``'sum'``, ``'mean'``, ``'max'``, or ``'attention'``. Default to 'mean'.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5,
                 emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1):
        super(GINPredictor, self).__init__()

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn = GIN(num_node_emb_list=num_node_emb_list,
                       num_edge_emb_list=num_edge_emb_list,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       JK=JK,
                       dropout=dropout)

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max' or 'attention', got {}".format(readout))

        if JK == 'concat':
            self.predict = nn.Linear((num_layers + 1) * emb_dim, n_tasks)
        else:
            self.predict = nn.Linear(emb_dim, n_tasks)

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(num_node_emb_list)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(g, categorical_node_feats, categorical_edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)
