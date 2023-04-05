"""
Gated Graph Neural Network module for graph classification tasks
"""
import torch

from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
from torch import nn


class GraphClsGGNN(nn.Module):
    def __init__(self, annotation_size, out_feats, n_steps, n_etypes, num_cls):
        super(GraphClsGGNN, self).__init__()

        self.annotation_size = annotation_size
        self.out_feats = out_feats

        self.ggnn = GatedGraphConv(
            in_feats=out_feats,
            out_feats=out_feats,
            n_steps=n_steps,
            n_etypes=n_etypes,
        )

        pooling_gate_nn = nn.Linear(annotation_size + out_feats, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn)
        self.output_layer = nn.Linear(annotation_size + out_feats, num_cls)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, graph, labels=None):
        etypes = graph.edata.pop("type")
        annotation = graph.ndata.pop("annotation").float()

        assert annotation.size()[-1] == self.annotation_size

        node_num = graph.num_nodes()

        zero_pad = torch.zeros(
            [node_num, self.out_feats - self.annotation_size],
            dtype=torch.float,
            device=annotation.device,
        )

        h1 = torch.cat([annotation, zero_pad], -1)
        out = self.ggnn(graph, h1, etypes)

        out = torch.cat([out, annotation], -1)

        out = self.pooling(graph, out)

        logits = self.output_layer(out)
        preds = torch.argmax(logits, -1)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, preds
        return preds
