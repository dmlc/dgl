"""Graph Isomorphism Networks."""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GIN']

# pylint: disable=W0221, C0103
class GINLayer(nn.Module):
    r"""Single Layer GIN from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    Parameters
    ----------
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    emb_dim : int
        The size of each embedding vector.
    batch_norm : bool
        Whether to apply batch normalization to the output of message passing.
        Default to True.
    activation : None or callable
        Activation function to apply to the output node representations.
        Default to None.
    """
    def __init__(self, num_edge_emb_list, emb_dim, batch_norm=True, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embeddings = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            nn.init.xavier_uniform_(emb_module.weight.data)
            self.edge_embeddings.append(emb_module)

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, g, node_feats, categorical_edge_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, emb_dim)
            * Input node features
            * N is the total number of nodes in the batch of graphs
            * emb_dim is the input node feature size, which must match emb_dim in initialization
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as len(self.edge_embeddings)
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        node_feats : float32 tensor of shape (N, emb_dim)
            Output node representations
        """
        edge_embeds = []
        for i, feats in enumerate(categorical_edge_feats):
            edge_embeds.append(self.edge_embeddings[i](feats))
        edge_embeds = torch.stack(edge_embeds, dim=0).sum(0)
        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embeds
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = g.ndata.pop('feat')
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats

class GIN(nn.Module):
    r"""Graph Isomorphism Network from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    This module is for updating node representations only.

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
        There can be four options for this argument, ``concat``, ``last``, ``max`` and ``sum``.
        Default to 'last'.

        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5
    """
    def __init__(self, num_node_emb_list, num_edge_emb_list,
                 num_layers=5, emb_dim=300, JK='last', dropout=0.5):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            nn.init.xavier_uniform_(emb_module.weight.data)
            self.node_embeddings.append(emb_module)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayer(num_edge_emb_list, emb_dim))
            else:
                self.gnn_layers.append(GINLayer(num_edge_emb_list, emb_dim, activation=F.relu))

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        """Update node representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(self.node_embeddings)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        final_node_feats : float32 tensor of shape (N, M)
            Output node representations, N for the number of nodes and
            M for output size. In particular, M will be emb_dim * (num_layers + 1)
            if self.JK == 'concat' and emb_dim otherwise.
        """
        node_embeds = []
        for i, feats in enumerate(categorical_node_feats):
            node_embeds.append(self.node_embeddings[i](feats))
        node_embeds = torch.stack(node_embeds, dim=0).sum(0)

        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](g, all_layer_node_feats[layer],
                                                categorical_edge_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze_(0) for h in all_layer_node_feats]
            final_node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze_(0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))

        return final_node_feats
