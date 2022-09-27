import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn as dglnn


class RelGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, rel_names):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.weight = {
            rel: nn.Parameter(th.Tensor(in_feat, out_feat))
            for rel in self.rel_names
        }
        for w in self.weight.values():
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))

    def forward(self, g, x_dict):
        h_dict = {ntype: 0 for ntype in x_dict.keys()}
        for stype, etype, dtype in g.canonical_etypes:
            h = x_dict[stype] @ self.weight[etype]  # dense mm
            h_dict[dtype] += g.adj[(stype, etype, dtype)] @ h
        h_dict = {ntype: th.relu(h) for ntype, h in h_dict.items()}
        return h_dict


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(
        self, g, embed_size, embed_name="embed", activation=None, dropout=0.0
    ):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(
                th.Tensor(g.number_of_nodes(ntype), self.embed_size)
            )
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation

        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds


class EntityClassify(nn.Module):
    def __init__(
        self,
        g,
        h_dim,
        out_dim,
        num_bases,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=False,
    ):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphEmbed(g, self.h_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayer(self.h_dim, self.h_dim, self.rel_names)
        )
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(self.h_dim, self.h_dim, self.rel_names)
            )
        # h2o
        self.layers.append(
            RelGraphConvLayer(self.h_dim, self.out_dim, self.rel_names)
        )

    def forward(self, h=None, blocks=None):
        if h is None:
            # full graph training
            h = self.embed_layer()
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h

    def inference(self, g, batch_size, device, num_workers, x=None):
        """Minibatch inference of final representation over all node types.

        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaningful.
        """

        if x is None:
            x = self.embed_layer()

        for l, layer in enumerate(self.layers):
            y = {
                k: th.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim,
                )
                for k in g.ntypes
            }

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g._graph,
                {k: th.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)

                h = {
                    k: x[k][input_nodes[k]].to(device)
                    for k in input_nodes.keys()
                }
                h = layer(block, h)

                for k in output_nodes.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y
