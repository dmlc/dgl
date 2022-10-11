import torch
import torch.nn as nn
from conv import GNN_node, GNN_node_Virtualnode

from dgl.nn.pytorch import (
    AvgPooling,
    GlobalAttentionPooling,
    MaxPooling,
    Set2Set,
    SumPooling,
)


class GNN(nn.Module):
    def __init__(
        self,
        num_tasks=1,
        num_layers=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0,
        JK="last",
        graph_pooling="sum",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNN_node(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumPooling()
        elif self.graph_pooling == "mean":
            self.pool = AvgPooling()
        elif self.graph_pooling == "max":
            self.pool = MaxPooling
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.BatchNorm1d(2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, 1),
                )
            )

        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, n_iters=2, n_layers=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, g, x, edge_attr):
        h_node = self.gnn_node(g, x, edge_attr)

        h_graph = self.pool(g, h_node)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=0, max=50)
