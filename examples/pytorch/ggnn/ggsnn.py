"""
Gated Graph Sequence Neural Network for sequence outputs
"""

import torch
import torch.nn.functional as F

from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
from torch import nn


class GGSNN(nn.Module):
    def __init__(
        self,
        annotation_size,
        out_feats,
        n_steps,
        n_etypes,
        max_seq_length,
        num_cls,
    ):
        super(GGSNN, self).__init__()

        self.annotation_size = annotation_size
        self.out_feats = out_feats
        self.max_seq_length = max_seq_length

        self.ggnn = GatedGraphConv(
            in_feats=out_feats,
            out_feats=out_feats,
            n_steps=n_steps,
            n_etypes=n_etypes,
        )

        self.annotation_out_layer = nn.Linear(
            annotation_size + out_feats, annotation_size
        )

        pooling_gate_nn = nn.Linear(annotation_size + out_feats, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn)

        self.output_layer = nn.Linear(annotation_size + out_feats, num_cls)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, graph, seq_lengths, ground_truth=None):
        etypes = graph.edata.pop("type")
        annotation = graph.ndata.pop("annotation").float()

        assert annotation.size()[-1] == self.annotation_size

        node_num = graph.num_nodes()

        all_logits = []
        for _ in range(self.max_seq_length):
            zero_pad = torch.zeros(
                [node_num, self.out_feats - self.annotation_size],
                dtype=torch.float,
                device=annotation.device,
            )

            h1 = torch.cat([annotation.detach(), zero_pad], -1)
            out = self.ggnn(graph, h1, etypes)
            out = torch.cat([out, annotation], -1)
            logits = self.pooling(graph, out)
            logits = self.output_layer(logits)
            all_logits.append(logits)

            annotation = self.annotation_out_layer(out)
            annotation = F.softmax(annotation, -1)

        all_logits = torch.stack(all_logits, 1)
        preds = torch.argmax(all_logits, -1)
        if ground_truth is not None:
            loss = sequence_loss(all_logits, ground_truth, seq_lengths)
            return loss, preds
        return preds


def sequence_loss(logits, ground_truth, seq_length=None):
    def sequence_mask(length):
        max_length = logits.size(1)
        batch_size = logits.size(0)
        range_tensor = torch.arange(
            0, max_length, dtype=seq_length.dtype, device=seq_length.device
        )
        range_tensor = torch.stack([range_tensor] * batch_size, 0)

        expanded_length = torch.stack([length] * max_length, -1)
        mask = (range_tensor < expanded_length).float()
        return mask

    loss = nn.CrossEntropyLoss(reduction="none")(
        logits.permute((0, 2, 1)), ground_truth
    )

    if seq_length is None:
        loss = loss.mean()
    else:
        mask = sequence_mask(seq_length)
        loss = (loss * mask).sum(-1) / seq_length.float()
        loss = loss.mean()
    return loss
