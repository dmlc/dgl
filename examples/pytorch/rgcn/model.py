from dgl import DGLGraph
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn.pytorch import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels,
                 regularizer="basis", num_bases=-1, dropout=0.,
                 self_loop=False, link_pred=False):
        super(RGCN, self).__init__()

        self.layers = nn.ModuleList()
        if link_pred:
            self.emb = nn.Embedding(in_dim, h_dim)
            in_dim = h_dim
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
    out_dev
        Device to store the output embeddings
    num_nodes : int
        Number of nodes in the graph.
    embed_size : int
        Output embed size
    """
    def __init__(self,
                 out_dev,
                 num_nodes,
                 embed_size):
        super(RelGraphEmbedLayer, self).__init__()
        self.out_dev = out_dev
        self.embed_size = embed_size

        # create embeddings for all nodes
        self.node_embed = nn.Embedding(num_nodes, embed_size, sparse=True)
        nn.init.uniform_(self.node_embed.weight, -1.0, 1.0)

    def forward(self, node_ids):
        """Forward computation

        Parameters
        ----------
        node_ids : tensor
            Raw node IDs.

        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = self.node_embed(node_ids).to(self.out_dev)

        return embeds
