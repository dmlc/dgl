import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ccorr


class CompGraphConv(nn.Module):
    """One layer of CompGCN."""

    def __init__(
        self, in_dim, out_dim, comp_fn="sub", batchnorm=True, dropout=0.1
    ):
        super(CompGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = th.tanh
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        # define in/out/loop transform layer
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)

        # define relation transform layer
        self.W_R = nn.Linear(self.in_dim, self.out_dim)

        # self loop embedding
        self.loop_rel = nn.Parameter(th.Tensor(1, self.in_dim))
        nn.init.xavier_normal_(self.loop_rel)

    def forward(self, g, n_in_feats, r_feats):
        with g.local_scope():
            # Assign values to source nodes. In a homogeneous graph, this is equal to
            # assigning them to all nodes.
            g.srcdata["h"] = n_in_feats
            # append loop_rel embedding to r_feats
            r_feats = th.cat((r_feats, self.loop_rel), 0)
            # Assign features to all edges with the corresponding relation embeddings
            g.edata["h"] = r_feats[g.edata["etype"]] * g.edata["norm"]

            # Compute composition function in 4 steps
            # Step 1: compute composition by edge in the edge direction, and store results in edges.
            if self.comp_fn == "sub":
                g.apply_edges(fn.u_sub_e("h", "h", out="comp_h"))
            elif self.comp_fn == "mul":
                g.apply_edges(fn.u_mul_e("h", "h", out="comp_h"))
            elif self.comp_fn == "ccorr":
                g.apply_edges(
                    lambda edges: {
                        "comp_h": ccorr(edges.src["h"], edges.data["h"])
                    }
                )
            else:
                raise Exception("Only supports sub, mul, and ccorr")

            # Step 2: use extracted edge direction to compute in and out edges
            comp_h = g.edata["comp_h"]

            in_edges_idx = th.nonzero(
                g.edata["in_edges_mask"], as_tuple=False
            ).squeeze()
            out_edges_idx = th.nonzero(
                g.edata["out_edges_mask"], as_tuple=False
            ).squeeze()

            comp_h_O = self.W_O(comp_h[out_edges_idx])
            comp_h_I = self.W_I(comp_h[in_edges_idx])

            new_comp_h = th.zeros(comp_h.shape[0], self.out_dim).to(
                comp_h.device
            )
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

            g.edata["new_comp_h"] = new_comp_h

            # Step 3: sum comp results to both src and dst nodes
            g.update_all(fn.copy_e("new_comp_h", "m"), fn.sum("m", "comp_edge"))

            # Step 4: add results of self-loop
            if self.comp_fn == "sub":
                comp_h_s = n_in_feats - r_feats[-1]
            elif self.comp_fn == "mul":
                comp_h_s = n_in_feats * r_feats[-1]
            elif self.comp_fn == "ccorr":
                comp_h_s = ccorr(n_in_feats, r_feats[-1])
            else:
                raise Exception("Only supports sub, mul, and ccorr")

            # Sum all of the comp results as output of nodes and dropout
            n_out_feats = (
                self.W_S(comp_h_s) + self.dropout(g.ndata["comp_edge"])
            ) * (1 / 3)

            # Compute relation output
            r_out_feats = self.W_R(r_feats)

            # Batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

        return n_out_feats, r_out_feats[:-1]


class CompGCN(nn.Module):
    def __init__(
        self,
        num_bases,
        num_rel,
        num_ent,
        in_dim=100,
        layer_size=[200],
        comp_fn="sub",
        batchnorm=True,
        dropout=0.1,
        layer_dropout=[0.3],
    ):
        super(CompGCN, self).__init__()

        self.num_bases = num_bases
        self.num_rel = num_rel
        self.num_ent = num_ent
        self.in_dim = in_dim
        self.layer_size = layer_size
        self.comp_fn = comp_fn
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.num_layer = len(layer_size)

        # CompGCN layers
        self.layers = nn.ModuleList()
        self.layers.append(
            CompGraphConv(
                self.in_dim,
                self.layer_size[0],
                comp_fn=self.comp_fn,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
            )
        )
        for i in range(self.num_layer - 1):
            self.layers.append(
                CompGraphConv(
                    self.layer_size[i],
                    self.layer_size[i + 1],
                    comp_fn=self.comp_fn,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                )
            )

        # Initial relation embeddings
        if self.num_bases > 0:
            self.basis = nn.Parameter(th.Tensor(self.num_bases, self.in_dim))
            self.weights = nn.Parameter(th.Tensor(self.num_rel, self.num_bases))
            nn.init.xavier_normal_(self.basis)
            nn.init.xavier_normal_(self.weights)
        else:
            self.rel_embds = nn.Parameter(th.Tensor(self.num_rel, self.in_dim))
            nn.init.xavier_normal_(self.rel_embds)

        # Node embeddings
        self.n_embds = nn.Parameter(th.Tensor(self.num_ent, self.in_dim))
        nn.init.xavier_normal_(self.n_embds)

        # Dropout after compGCN layers
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(nn.Dropout(self.layer_dropout[i]))

    def forward(self, graph):
        # node and relation features
        n_feats = self.n_embds
        if self.num_bases > 0:
            r_embds = th.mm(self.weights, self.basis)
            r_feats = r_embds
        else:
            r_feats = self.rel_embds

        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(graph, n_feats, r_feats)
            n_feats = dropout(n_feats)

        return n_feats, r_feats


# Use convE as the score function
class CompGCN_ConvE(nn.Module):
    def __init__(
        self,
        num_bases,
        num_rel,
        num_ent,
        in_dim,
        layer_size,
        comp_fn="sub",
        batchnorm=True,
        dropout=0.1,
        layer_dropout=[0.3],
        num_filt=200,
        hid_drop=0.3,
        feat_drop=0.3,
        ker_sz=5,
        k_w=5,
        k_h=5,
    ):
        super(CompGCN_ConvE, self).__init__()

        self.embed_dim = layer_size[-1]
        self.hid_drop = hid_drop
        self.feat_drop = feat_drop
        self.ker_sz = ker_sz
        self.k_w = k_w
        self.k_h = k_h
        self.num_filt = num_filt

        # compGCN model to get sub/rel embs
        self.compGCN_Model = CompGCN(
            num_bases,
            num_rel,
            num_ent,
            in_dim,
            layer_size,
            comp_fn,
            batchnorm,
            dropout,
            layer_dropout,
        )

        # batchnorms to the combined (sub+rel) emb
        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn1 = th.nn.BatchNorm2d(self.num_filt)
        self.bn2 = th.nn.BatchNorm1d(self.embed_dim)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = th.nn.Dropout(self.hid_drop)
        self.feature_drop = th.nn.Dropout(self.feat_drop)
        self.m_conv1 = th.nn.Conv2d(
            1,
            out_channels=self.num_filt,
            kernel_size=(self.ker_sz, self.ker_sz),
            stride=1,
            padding=0,
            bias=False,
        )

        flat_sz_h = int(2 * self.k_w) - self.ker_sz + 1
        flat_sz_w = self.k_h - self.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = th.nn.Linear(self.flat_sz, self.embed_dim)

        # bias to the score
        self.bias = nn.Parameter(th.zeros(num_ent))

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def forward(self, graph, sub, rel):
        # get sub_emb and rel_emb via compGCN
        n_feats, r_feats = self.compGCN_Model(graph)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]

        # combine the sub_emb and rel_emb
        stk_inp = self.concat(sub_emb, rel_emb)
        # use convE to score the combined emb
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # compute score
        x = th.mm(x, n_feats.transpose(1, 0))
        # add in bias
        x += self.bias.expand_as(x)
        score = th.sigmoid(x)
        return score
