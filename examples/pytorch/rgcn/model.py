import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 regularizer="basis", num_bases=-1, dropout=0.,
                 self_loop=False, link_pred=False):
        super(RGCN, self).__init__()

        self.layers = nn.ModuleList()
        if link_pred:
            self.emb = nn.Embedding(num_nodes, h_dim)
        else:
            self.emb = None
            self.layers.append(RelGraphConv(num_nodes, h_dim, num_rels, regularizer,
                                            num_bases, activation=F.relu, self_loop=self_loop,
                                            dropout=dropout))

        # For entity classification, dropout should not be applied to the output layer
        if not link_pred:
            dropout = 0.
        self.layers.append(RelGraphConv(h_dim, out_dim, num_rels, regularizer,
                                        num_bases, self_loop=self_loop, dropout=dropout))

    def forward(self, g, h, r, norm):
        if self.emb is not None:
            h = self.emb(h.squeeze())

        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    storage_dev_id : int
        The device to store the weights of the layer.
    out_dev_id : int
        Device to return the output embeddings on.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 storage_dev_id,
                 out_dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.storage_dev_id = th.device( \
            storage_dev_id if storage_dev_id >= 0 else 'cpu')
        self.out_dev_id = th.device(out_dev_id if out_dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer, device=self.storage_dev_id)
                else:
                    sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    sparse_emb.cuda(self.storage_dev_id)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.empty([input_emb_size, self.embed_size],
                                              device=self.storage_dev_id))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

    @property
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.out_dev_id)

        # transfer input to the correct device
        type_ids = type_ids.to(self.storage_dev_id)
        node_tids = node_tids.to(self.storage_dev_id)

        # build locs first
        locs = [None for i in range(self.num_of_ntype)]
        for ntype in range(self.num_of_ntype):
            locs[ntype] = (node_tids == ntype).nonzero().squeeze(-1)
        for ntype in range(self.num_of_ntype):
            loc = locs[ntype]
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.out_dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.out_dev_id)
            else:
                embeds[loc] = features[ntype][type_ids[loc]].to(self.out_dev_id) @ self.embeds[str(ntype)].to(self.out_dev_id)

        return embeds
