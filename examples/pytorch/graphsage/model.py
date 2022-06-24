import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn.functional as F
import tqdm
# below import for distributed/multi-gpu training examples
from dgl.multiprocessing import shared_tensor
import torch.distributed as dist

# parameters can be tuned as below, or as keyword arguments when create the model
activation = F.relu
dropout = 0.5

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 n_hidden,
                 n_layers,
                 aggregator_type):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
            # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class GraphSAGEBatch(GraphSAGE):
    def __init__(self,
                 in_feats,
                 n_classes,
                 n_hidden,
                 n_layers,
                 aggregator_type):
        super().__init__(in_feats, n_classes, n_hidden, n_layers, aggregator_type)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """
        Perform inference in layer-major order rather than batch-major order.
        That is, infer the first layer for the entire graph, and store the
        intermediate values h_0, before infering the second layer to generate
        h_1. This is done for two reasons: 1) it limits the effect of node
        degree on the amount of memory used as it only proccesses 1-hop
        neighbors at a time, and 2) it reduces the total amount of computation
        required as each node is only processed once per layer.

        Parameters
        ----------
            g : DGLGraph
                The graph to perform inference on.
            device : context
                The device this process should use for inference
            batch_size : int
                The number of items to collect in a batch.

        Returns
        -------
            tensor
                The predictions for all nodes in the graph.
        """
        feat = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device='cpu', pin_memory=True)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # use an explicitly contiguous slice
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous so we can take
                # advantage of that here
                y[output_nodes[0]:output_nodes[-1]+1] = h.to('cpu')
            feat = y
        return y

class GraphSAGEBatchMultiGPU(GraphSAGEBatch):
     """
        Perform inference in layer-major order rather than batch-major order within the context of multi-gpu training.
     """
     def inference(self, g, device, batch_size):
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        use_uva = (g.device != device)

        for l, layer in enumerate(self.layers):
            """
               Have to re-initialize dataloader for every layer to allow tensor pinned by DGL being automatically unpinned
            """
            dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes(), device=device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0, use_ddp=True, use_uva=use_uva)
             
            # in order to prevent running out of GPU memory, we allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes))

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader) \
                    if dist.get_rank() == 0 else dataloader:
                x = blocks[0].srcdata['h']
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(y.device)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            if l + 1 < len(self.layers):
                # assign the output features of this layer as the new input
                # features for the next layer
                if use_uva:
                    g.ndata['h'] = y
                else:
                    g.ndata['h'] = y.to(g.device)
            else:
                # remove the intermediate data from the graph
                g.ndata.pop('h')
        return y
