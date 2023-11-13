import dgl.sparse as dglsp
import torch.nn as nn
import torch.nn.functional as F

from utils import LinearNeuralNetwork


class OGC(nn.Module):
    def __init__(self, graph):
        super(OGC, self).__init__()
        self.linear_clf = LinearNeuralNetwork(
            nfeat=graph.ndata["feat"].shape[1],
            nclass=graph.ndata["label"].max().item() + 1,
            bias=False,
        )

        self.label = graph.ndata["label"]
        self.label_one_hot = F.one_hot(graph.ndata["label"]).float()
        # LIM trick, else use both train and val set to construct this matrix.
        self.label_idx_mat = dglsp.diag(graph.ndata["train_mask"]).float()

        self.test_mask = graph.ndata["test_mask"]
        self.tv_mask = graph.ndata["train_mask"] + graph.ndata["val_mask"]

    def forward(self, x):
        return self.linear_clf(x)

    def update_embeds(self, embeds, lazy_adj, args):
        """Update classifier's weight by training a linear supervised model."""
        pred_label = self(embeds).data
        clf_weight = self.linear_clf.W.weight.data

        # Update the smoothness loss via LGC.
        embeds = dglsp.spmm(lazy_adj, embeds)

        # Update the supervised loss via SEB.
        deriv_sup = 2 * dglsp.matmul(
            dglsp.spmm(self.label_idx_mat, -self.label_one_hot + pred_label),
            clf_weight,
        )
        embeds = embeds - args.lr_sup * deriv_sup

        args.lr_sup = args.lr_sup * args.decline
        return embeds
