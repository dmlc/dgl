import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GINConv, NNConv, Set2Set
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import BatchNorm1d, GRU, Linear, ModuleList, ReLU, Sequential
from utils import global_global_loss_, local_global_loss_

""" Feedforward neural network"""


class FeedforwardNetwork(nn.Module):

    """
    3-layer feed-forward neural networks with jumping connections
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(feat):
        feat: Tensor
            [N * D], input features
    """

    def __init__(self, in_dim, hid_dim):
        super(FeedforwardNetwork, self).__init__()

        self.block = Sequential(
            Linear(in_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU(),
        )

        self.jump_con = Linear(in_dim, hid_dim)

    def forward(self, feat):
        block_out = self.block(feat)
        jump_out = self.jump_con(feat)

        out = block_out + jump_out

        return out


""" Unsupervised Setting """


class GINEncoder(nn.Module):
    """
    Encoder based on dgl.nn.GINConv &  dgl.nn.SumPooling
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    n_layer:
        Number of GIN layers.

    Functions
    -----------
    forward(graph, feat):
        graph: DGLGraph
        feat: Tensor
            [N * D], node features
    """

    def __init__(self, in_dim, hid_dim, n_layer):
        super(GINEncoder, self).__init__()

        self.n_layer = n_layer

        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(n_layer):
            if i == 0:
                n_in = in_dim
            else:
                n_in = hid_dim
            n_out = hid_dim
            block = Sequential(
                Linear(n_in, n_out), ReLU(), Linear(hid_dim, hid_dim)
            )

            conv = GINConv(apply_func=block, aggregator_type="sum")
            bn = BatchNorm1d(hid_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        # sum pooling
        self.pool = SumPooling()

    def forward(self, graph, feat):
        xs = []
        x = feat
        for i in range(self.n_layer):
            x = F.relu(self.convs[i](graph, x))
            x = self.bns[i](x)
            xs.append(x)

        local_emb = th.cat(xs, 1)  # patch-level embedding
        global_emb = self.pool(graph, local_emb)  # graph-level embedding

        return global_emb, local_emb


class InfoGraph(nn.Module):
    r"""
        InfoGraph model for unsupervised setting

    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    n_layer: int
        Number of the GNN encoder layers.

    Functions
    -----------
    forward(graph):
        graph: DGLGraph

    """

    def __init__(self, in_dim, hid_dim, n_layer):
        super(InfoGraph, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.n_layer = n_layer
        embedding_dim = hid_dim * n_layer

        self.encoder = GINEncoder(in_dim, hid_dim, n_layer)

        self.local_d = FeedforwardNetwork(
            embedding_dim, embedding_dim
        )  # local discriminator (node-level)
        self.global_d = FeedforwardNetwork(
            embedding_dim, embedding_dim
        )  # global discriminator (graph-level)

    def get_embedding(self, graph, feat):
        # get_embedding function for evaluation the learned embeddings

        with th.no_grad():
            global_emb, _ = self.encoder(graph, feat)

        return global_emb

    def forward(self, graph, feat, graph_id):
        global_emb, local_emb = self.encoder(graph, feat)

        global_h = self.global_d(global_emb)  # global hidden representation
        local_h = self.local_d(local_emb)  # local hidden representation

        loss = local_global_loss_(local_h, global_h, graph_id)

        return loss


""" Semisupervised Setting """


class NNConvEncoder(nn.Module):

    """
    Encoder based on dgl.nn.NNConv & GRU & dgl.nn.set2set pooling
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(graph, nfeat, efeat):
        graph: DGLGraph
        nfeat: Tensor
            [N * D1], node features
        efeat: Tensor
            [E * D2], edge features
    """

    def __init__(self, in_dim, hid_dim):
        super(NNConvEncoder, self).__init__()

        self.lin0 = Linear(in_dim, hid_dim)

        # mlp for edge convolution in NNConv
        block = Sequential(
            Linear(5, 128), ReLU(), Linear(128, hid_dim * hid_dim)
        )

        self.conv = NNConv(
            hid_dim,
            hid_dim,
            edge_func=block,
            aggregator_type="mean",
            residual=False,
        )
        self.gru = GRU(hid_dim, hid_dim)

        # set2set pooling
        self.set2set = Set2Set(hid_dim, n_iters=3, n_layers=1)

    def forward(self, graph, nfeat, efeat):
        out = F.relu(self.lin0(nfeat))
        h = out.unsqueeze(0)

        feat_map = []

        # Convolution layer number is 3
        for i in range(3):
            m = F.relu(self.conv(graph, out, efeat))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map.append(out)

        out = self.set2set(graph, out)

        # out: global embedding, feat_map[-1]: local embedding
        return out, feat_map[-1]


class InfoGraphS(nn.Module):

    """
    InfoGraph* model for semi-supervised setting
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(graph):
        graph: DGLGraph

    unsupforward(graph):
        graph: DGLGraph

    """

    def __init__(self, in_dim, hid_dim):
        super(InfoGraphS, self).__init__()

        self.sup_encoder = NNConvEncoder(in_dim, hid_dim)
        self.unsup_encoder = NNConvEncoder(in_dim, hid_dim)

        self.fc1 = Linear(2 * hid_dim, hid_dim)
        self.fc2 = Linear(hid_dim, 1)

        # unsupervised local discriminator and global discriminator for local-global infomax
        self.unsup_local_d = FeedforwardNetwork(hid_dim, hid_dim)
        self.unsup_global_d = FeedforwardNetwork(2 * hid_dim, hid_dim)

        # supervised global discriminator and unsupervised global discriminator for global-global infomax
        self.sup_d = FeedforwardNetwork(2 * hid_dim, hid_dim)
        self.unsup_d = FeedforwardNetwork(2 * hid_dim, hid_dim)

    def forward(self, graph, nfeat, efeat):
        sup_global_emb, sup_local_emb = self.sup_encoder(graph, nfeat, efeat)

        sup_global_pred = self.fc2(F.relu(self.fc1(sup_global_emb)))
        sup_global_pred = sup_global_pred.view(-1)

        return sup_global_pred

    def unsup_forward(self, graph, nfeat, efeat, graph_id):
        sup_global_emb, sup_local_emb = self.sup_encoder(graph, nfeat, efeat)
        unsup_global_emb, unsup_local_emb = self.unsup_encoder(
            graph, nfeat, efeat
        )

        g_enc = self.unsup_global_d(unsup_global_emb)
        l_enc = self.unsup_local_d(unsup_local_emb)

        sup_g_enc = self.sup_d(sup_global_emb)
        unsup_g_enc = self.unsup_d(unsup_global_emb)

        # Calculate loss
        unsup_loss = local_global_loss_(l_enc, g_enc, graph_id)
        con_loss = global_global_loss_(sup_g_enc, unsup_g_enc)

        return unsup_loss, con_loss
