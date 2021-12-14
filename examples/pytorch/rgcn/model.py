from dgl import DGLGraph
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn.pytorch import RelGraphConv

class RGCN(nn.Module):
    """
    Parameters
    ----------
    in_dim : int
        Input feature size or number of nodes
    """
    def __init__(self, in_dim, h_dim, out_dim, num_rels,
                 regularizer="basis", num_bases=-1, dropout=0.,
                 self_loop=False, link_pred=False):
        super(RGCN, self).__init__()

        self.layers = nn.ModuleList()
        if link_pred:
            self.emb = nn.Embedding(in_dim, h_dim)
        else:
            self.emb = None
            self.layers.append(RelGraphConv(in_dim, h_dim, num_rels, regularizer,
                                            num_bases, activation=F.relu, self_loop=self_loop,
                                            dropout=dropout))

        # For entity classification, dropout should not be applied to the output layer
        if not link_pred:
            dropout = 0.
        self.layers.append(RelGraphConv(h_dim, out_dim, num_rels, regularizer,
                                        num_bases, self_loop=self_loop, dropout=dropout))

    def forward(self, g, h):
        if isinstance(g, DGLGraph):
            blocks = [g] * len(self.layers)
        else:
            blocks = g

        if self.emb is not None:
            h = self.emb(h.squeeze())

        for layer, block in zip(self.layers, blocks):
            h = layer(block, h, block.edata[dgl.ETYPE], block.edata['norm'])
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    """Embedding layer for featureless heterograph.

    Parameters
    ----------
    storage_dev
        Device to store the weights of the layer
    out_dev
        Device to store the output embeddings
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as a one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding, otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 storage_dev,
                 out_dev,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.storage_dev_id = storage_dev
        self.out_dev_id = out_dev
        self.embed_size = embed_size
        self.dgl_sparse = dgl_sparse

        # create embeddings for all nodes
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        num_of_ntype = len(input_size)
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if dgl_sparse:
                self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                    init_func=initializer, device=self.storage_dev_id)
            else:
                sparse_emb = nn.Embedding(input_size[ntype], embed_size, sparse=True)
                sparse_emb.cuda(self.storage_dev_id)
                nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                self.node_embeds[str(ntype)] = sparse_emb

    @property
    def dgl_emb(self):
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids):
        """Forward computation

        Parameters
        ----------
        node_tids : tensor
            node type ids
        type_ids : tensor
            type-specific node ids

        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = th.empty(node_tids.shape[0], self.embed_size, device=self.out_dev_id)

        type_ids = type_ids.to(self.storage_dev_id)
        node_tids = node_tids.to(self.storage_dev_id)

        # build locs first
        locs = [None for _ in range(self.num_of_ntype)]
        for ntype in range(self.num_of_ntype):
            locs[ntype] = (node_tids == ntype).nonzero().squeeze(-1)
        for ntype in range(self.num_of_ntype):
            loc = locs[ntype]
            if self.dgl_sparse:
                embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.out_dev_id)
            else:
                embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.out_dev_id)

        return embeds
