import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, SAGEConv, SortPooling, SumPooling


class GCN(nn.Module):
    """
    GCN Model

    Attributes:
        num_layers(int): num of gcn layers
        hidden_units(int): num of hidden units
        gcn_type(str): type of gcn layer, 'gcn' for GraphConv and 'sage' for SAGEConv
        pooling_type(str): type of graph pooling to get subgraph representation
                           'sum' for sum pooling and 'center' for center pooling.
        node_attributes(Tensor, optional): node attribute
        edge_weights(Tensor, optional): edge weight
        node_embedding(Tensor, optional): pre-trained node embedding
        use_embedding(bool, optional): whether to use node embedding. Note that if 'use_embedding' is set True
                             and 'node_embedding' is None, will automatically randomly initialize node embedding.
        num_nodes(int, optional): num of nodes
        dropout(float, optional): dropout rate
        max_z(int, optional): default max vocab size of node labeling, default 1000.

    """

    def __init__(
        self,
        num_layers,
        hidden_units,
        gcn_type="gcn",
        pooling_type="sum",
        node_attributes=None,
        edge_weights=None,
        node_embedding=None,
        use_embedding=False,
        num_nodes=None,
        dropout=0.5,
        max_z=1000,
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.use_attribute = False if node_attributes is None else True
        self.use_embedding = use_embedding
        self.use_edge_weight = False if edge_weights is None else True

        self.z_embedding = nn.Embedding(max_z, hidden_units)
        if node_attributes is not None:
            self.node_attributes_lookup = nn.Embedding.from_pretrained(
                node_attributes
            )
            self.node_attributes_lookup.weight.requires_grad = False
        if edge_weights is not None:
            self.edge_weights_lookup = nn.Embedding.from_pretrained(
                edge_weights
            )
            self.edge_weights_lookup.weight.requires_grad = False
        if node_embedding is not None:
            self.node_embedding = nn.Embedding.from_pretrained(node_embedding)
            self.node_embedding.weight.requires_grad = False
        elif use_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_units)

        initial_dim = hidden_units
        if self.use_attribute:
            initial_dim += self.node_attributes_lookup.embedding_dim
        if self.use_embedding:
            initial_dim += self.node_embedding.embedding_dim

        self.layers = nn.ModuleList()
        if gcn_type == "gcn":
            self.layers.append(
                GraphConv(initial_dim, hidden_units, allow_zero_in_degree=True)
            )
            for _ in range(num_layers - 1):
                self.layers.append(
                    GraphConv(
                        hidden_units, hidden_units, allow_zero_in_degree=True
                    )
                )
        elif gcn_type == "sage":
            self.layers.append(
                SAGEConv(initial_dim, hidden_units, aggregator_type="gcn")
            )
            for _ in range(num_layers - 1):
                self.layers.append(
                    SAGEConv(hidden_units, hidden_units, aggregator_type="gcn")
                )
        else:
            raise ValueError("Gcn type error.")

        self.linear_1 = nn.Linear(hidden_units, hidden_units)
        self.linear_2 = nn.Linear(hidden_units, 1)
        if pooling_type != "sum":
            raise ValueError("Pooling type error.")
        self.pooling = SumPooling()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, z, node_id=None, edge_id=None):
        """
        Args:
            g(DGLGraph): the graph
            z(Tensor): node labeling tensor, shape [N, 1]
            node_id(Tensor, optional): node id tensor, shape [N, 1]
            edge_id(Tensor, optional): edge id tensor, shape [E, 1]
        Returns:
            x(Tensor): output tensor

        """

        z_emb = self.z_embedding(z)

        if self.use_attribute:
            x = self.node_attributes_lookup(node_id)
            x = torch.cat([z_emb, x], 1)
        else:
            x = z_emb

        if self.use_edge_weight:
            edge_weight = self.edge_weights_lookup(edge_id)
        else:
            edge_weight = None

        if self.use_embedding:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)

        for layer in self.layers[:-1]:
            x = layer(g, x, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x, edge_weight=edge_weight)

        x = self.pooling(g, x)
        x = F.relu(self.linear_1(x))
        F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)

        return x


class DGCNN(nn.Module):
    """
    An end-to-end deep learning architecture for graph classification.
    paper link: https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf

    Attributes:
        num_layers(int): num of gcn layers
        hidden_units(int): num of hidden units
        k(int, optional): The number of nodes to hold for each graph in SortPooling.
        gcn_type(str): type of gcn layer, 'gcn' for GraphConv and 'sage' for SAGEConv
        node_attributes(Tensor, optional): node attribute
        edge_weights(Tensor, optional): edge weight
        node_embedding(Tensor, optional): pre-trained node embedding
        use_embedding(bool, optional): whether to use node embedding. Note that if 'use_embedding' is set True
                             and 'node_embedding' is None, will automatically randomly initialize node embedding.
        num_nodes(int, optional): num of nodes
        dropout(float, optional): dropout rate
        max_z(int, optional): default max vocab size of node labeling, default 1000.
    """

    def __init__(
        self,
        num_layers,
        hidden_units,
        k=10,
        gcn_type="gcn",
        node_attributes=None,
        edge_weights=None,
        node_embedding=None,
        use_embedding=False,
        num_nodes=None,
        dropout=0.5,
        max_z=1000,
    ):
        super(DGCNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attribute = False if node_attributes is None else True
        self.use_embedding = use_embedding
        self.use_edge_weight = False if edge_weights is None else True

        self.z_embedding = nn.Embedding(max_z, hidden_units)

        if node_attributes is not None:
            self.node_attributes_lookup = nn.Embedding.from_pretrained(
                node_attributes
            )
            self.node_attributes_lookup.weight.requires_grad = False
        if edge_weights is not None:
            self.edge_weights_lookup = nn.Embedding.from_pretrained(
                edge_weights
            )
            self.edge_weights_lookup.weight.requires_grad = False
        if node_embedding is not None:
            self.node_embedding = nn.Embedding.from_pretrained(node_embedding)
            self.node_embedding.weight.requires_grad = False
        elif use_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_units)

        initial_dim = hidden_units
        if self.use_attribute:
            initial_dim += self.node_attributes_lookup.embedding_dim
        if self.use_embedding:
            initial_dim += self.node_embedding.embedding_dim

        self.layers = nn.ModuleList()
        if gcn_type == "gcn":
            self.layers.append(
                GraphConv(initial_dim, hidden_units, allow_zero_in_degree=True)
            )
            for _ in range(num_layers - 1):
                self.layers.append(
                    GraphConv(
                        hidden_units, hidden_units, allow_zero_in_degree=True
                    )
                )
            self.layers.append(
                GraphConv(hidden_units, 1, allow_zero_in_degree=True)
            )
        elif gcn_type == "sage":
            self.layers.append(
                SAGEConv(initial_dim, hidden_units, aggregator_type="gcn")
            )
            for _ in range(num_layers - 1):
                self.layers.append(
                    SAGEConv(hidden_units, hidden_units, aggregator_type="gcn")
                )
            self.layers.append(SAGEConv(hidden_units, 1, aggregator_type="gcn"))
        else:
            raise ValueError("Gcn type error.")

        self.pooling = SortPooling(k=k)
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_units * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv_1 = nn.Conv1d(
            1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0]
        )
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv_2 = nn.Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1
        )
        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.linear_1 = nn.Linear(dense_dim, 128)
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, g, z, node_id=None, edge_id=None):
        """
        Args:
            g(DGLGraph): the graph
            z(Tensor): node labeling tensor, shape [N, 1]
            node_id(Tensor, optional): node id tensor, shape [N, 1]
            edge_id(Tensor, optional): edge id tensor, shape [E, 1]
        Returns:
            x(Tensor): output tensor
        """
        z_emb = self.z_embedding(z)
        if self.use_attribute:
            x = self.node_attributes_lookup(node_id)
            x = torch.cat([z_emb, x], 1)
        else:
            x = z_emb
        if self.use_edge_weight:
            edge_weight = self.edge_weights_lookup(edge_id)
        else:
            edge_weight = None

        if self.use_embedding:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)

        xs = [x]
        for layer in self.layers:
            out = torch.tanh(layer(g, xs[-1], edge_weight=edge_weight))
            xs += [out]

        x = torch.cat(xs[1:], dim=-1)

        # SortPooling
        x = self.pooling(g, x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv_1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv_2(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear_1(x))
        F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)

        return x
