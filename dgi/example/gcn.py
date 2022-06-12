import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows
from dgi.utils import update_out_in_chunks

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, n_layer, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.convs = nn.ModuleList()
        self.n_layers = n_layer
        if n_layer == 1:
            self.convs.append(dgl.nn.GraphConv(in_features, out_features))
        else:
            self.convs.append(dgl.nn.GraphConv(in_features, hidden_features))
            for l in range(n_layer - 2):
                self.convs.append(dgl.nn.GraphConv(hidden_features, hidden_features))
            self.convs.append(dgl.nn.GraphConv(hidden_features, out_features))

    def forward(self, blocks, x):
        for i, conv in enumerate(self.convs):
            x_dst = x[:blocks[i].number_of_dst_nodes()]
            x = F.relu(conv(blocks[i], (x, x_dst)))
        return x

    def forward_full(self, blocks, x):
        for conv in self.convs:
            x_dst = x[:blocks.number_of_dst_nodes()]
            x = F.relu(conv(blocks, (x, x_dst)))
        return x

    def inference(self, g, batch_size, device, x, nids, use_uva = False, use_ssd = False):
        if use_uva:
            for k in list(g.ndata.keys()):
                g.ndata.pop(k)
            for k in list(g.edata.keys()):
                g.edata.pop(k)

        # Compute representations layer by layer
        for l, layer in enumerate(self.convs):
            shape = (g.number_of_nodes(),
                            self.hidden_features
                            if l != self.n_layers - 1
                            else self.out_features)
            if use_ssd:
                y = torch.as_tensor(np.memmap(f"/ssd/feat_{l}.npy",dtype=np.float32, mode="w+", shape=shape, ))
            else:
                y = torch.zeros(*shape)

            if use_uva:
                pin_memory_inplace(x)
                nids = nids.to(device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, nids, sampler,
                batch_size=batch_size,
                shuffle=False,
                use_uva=use_uva,
                drop_last=False,
                device=torch.device('cuda'),
                num_workers=0)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                torch.cuda.empty_cache()
                block = blocks[0].to(device)

                # Copy the features of necessary input nodes to GPU
                if use_uva:
                    h = gather_pinned_tensor_rows(x, input_nodes)
                else:
                    h = x[input_nodes].to(device)
                # Compute output.  Note that this computation is the same
                # but only for a single layer.
                h_dst = h[:block.number_of_dst_nodes()]
                h = F.relu(layer(block, (h, h_dst)))
                # Copy to output back to CPU.
                update_out_in_chunks(y, output_nodes, h)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            if use_uva:
                unpin_memory_inplace(x)
            x = y
        return y
