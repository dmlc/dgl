import torch.nn as nn
import dgl
from dgl.base import dgl_warning
class GraphSAGE(nn.Module):
    def __init__(self,
                 data_info: dict,
                 embed_size: int = -1,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 activation: str = "relu",
                 dropout: float = 0.5,
                 aggregator_type: str = "gcn"):        
        """GraphSAGE model

        Parameters
        ----------
        data_info : dict
            the information about the input dataset. Should contain feilds "in_size", "out_size", "num_nodes"
        embed_size : int
            The dimension of created embedding table. -1 means using original node embedding
        hidden_size : int
            Hidden size.
        num_layers : int
            Number of hidden layers.
        dropout : float
            Dropout rate.
        aggregator_type : str
            Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
        """
        super(GraphSAGE, self).__init__()
        self.data_info = data_info
        self.out_size = data_info["out_size"]
        self.in_size = data_info["in_size"]
        self.embed_size = embed_size
        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn.functional, activation)

        # input layer
        self.layers.append(dgl.nn.SAGEConv(self.in_size, hidden_size, aggregator_type))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(hidden_size, hidden_size, aggregator_type))
        # output layer
        self.layers.append(dgl.nn.SAGEConv(hidden_size, self.out_size, aggregator_type)) # activation None

    def forward(self, graph, node_feat, edge_feat = None):
        if self.embed_size > 0:
            dgl_warning("The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size.", norepeat=True)
            h = self.embed.weight
        else:
            h = node_feat
        h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
    
    def forward_block(self, blocks, node_feat, edge_feat = None):
        h = node_feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
