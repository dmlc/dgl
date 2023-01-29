"""
SudokuNN module based on RRN for solving sudoku puzzles
"""

import torch
from rrn import RRN
from torch import nn


class SudokuNN(nn.Module):
    def __init__(self, num_steps, embed_size=16, hidden_dim=96, edge_drop=0.1):
        super(SudokuNN, self).__init__()
        self.num_steps = num_steps

        self.digit_embed = nn.Embedding(10, embed_size)
        self.row_embed = nn.Embedding(9, embed_size)
        self.col_embed = nn.Embedding(9, embed_size)

        self.input_layer = nn.Sequential(
            nn.Linear(3 * embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lstm = nn.LSTMCell(hidden_dim * 2, hidden_dim, bias=False)

        msg_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rrn = RRN(msg_layer, self.node_update_func, num_steps, edge_drop)

        self.output_layer = nn.Linear(hidden_dim, 10)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, g, is_training=True):
        labels = g.ndata.pop("a")

        input_digits = self.digit_embed(g.ndata.pop("q"))
        rows = self.row_embed(g.ndata.pop("row"))
        cols = self.col_embed(g.ndata.pop("col"))

        x = self.input_layer(torch.cat([input_digits, rows, cols], -1))
        g.ndata["x"] = x
        g.ndata["h"] = x
        g.ndata["rnn_h"] = torch.zeros_like(x, dtype=torch.float)
        g.ndata["rnn_c"] = torch.zeros_like(x, dtype=torch.float)

        outputs = self.rrn(g, is_training)
        logits = self.output_layer(outputs)

        preds = torch.argmax(logits, -1)

        if is_training:
            labels = torch.stack([labels] * self.num_steps, 0)
        logits = logits.view([-1, 10])
        labels = labels.view([-1])
        loss = self.loss_func(logits, labels)
        return preds, loss

    def node_update_func(self, nodes):
        x, h, m, c = (
            nodes.data["x"],
            nodes.data["rnn_h"],
            nodes.data["m"],
            nodes.data["rnn_c"],
        )
        new_h, new_c = self.lstm(torch.cat([x, m], -1), (h, c))
        return {"h": new_h, "rnn_c": new_c, "rnn_h": new_h}
