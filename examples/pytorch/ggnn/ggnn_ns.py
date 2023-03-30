"""
Gated Graph Neural Network module for node selection tasks
"""
import dgl
import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn


class NodeSelectionGGNN(nn.Module):
    def __init__(self, annotation_size, out_feats, n_steps, n_etypes):
        super(NodeSelectionGGNN, self).__init__()

        self.annotation_size = annotation_size
        self.out_feats = out_feats

        self.ggnn = GatedGraphConv(
            in_feats=out_feats,
            out_feats=out_feats,
            n_steps=n_steps,
            n_etypes=n_etypes,
        )

        self.output_layer = nn.Linear(annotation_size + out_feats, 1)
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

        all_logits = self.output_layer(
            torch.cat([out, annotation], -1)
        ).squeeze(-1)
        graph.ndata["logits"] = all_logits

        batch_g = dgl.unbatch(graph)

        preds = []
        if labels is not None:
            loss = 0.0
        for i, g in enumerate(batch_g):
            logits = g.ndata["logits"]
            preds.append(torch.argmax(logits))
            if labels is not None:
                logits = logits.unsqueeze(0)
                y = labels[i].unsqueeze(0)
                loss += self.loss_fn(logits, y)

        if labels is not None:
            loss /= float(len(batch_g))
            return loss, preds
        return preds
