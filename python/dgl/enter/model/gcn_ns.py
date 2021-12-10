import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm

class GCN(nn.Module):
    def __init__(self, in_size,
                 out_size,
                 hidden_size: int = 16,
                 num_layers: int = 2,
                 activation = F.relu,
                 dropout: float = 0.5
                 ):
        super().__init__()
        self.init(in_size, hidden_size, out_size, num_layers, activation, dropout)

    def init(self, in_size, hidden_size, out_size, num_layers, activation, dropout):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_size, hidden_size, 'mean'))
            for i in range(1, num_layers - 1):
                self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hidden_size, out_size, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def forward_block

    def forward

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.hidden_size if l != len(self.layers) - 1 else self.out_size)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y