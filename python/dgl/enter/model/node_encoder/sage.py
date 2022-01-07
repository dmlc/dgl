import torch.nn as nn
import dgl

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 activation: str = "relu",
                 dropout: float = 0.5,
                 aggregator_type: str = "gcn"):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn.functional, activation)

        # input layer
        self.layers.append(dgl.nn.SAGEConv(in_size, hidden_size, aggregator_type))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(hidden_size, hidden_size, aggregator_type))
        # output layer
        self.layers.append(dgl.nn.SAGEConv(hidden_size, out_size, aggregator_type)) # activation None

    def forward(self, graph, inputs, edge_feat = None):
        h = self.dropout(inputs)
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
