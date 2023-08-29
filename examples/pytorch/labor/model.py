import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch as th
import torch.functional as F
import torch.nn as nn

from dgl.nn import GATv2Conv


class GATv2(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                True,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    True,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                True,
                bias=False,
                share_weights=True,
            )
        )

    def forward(self, mfgs, h):
        for l, mfg in enumerate(mfgs):
            h = self.gatv2_layers[l](mfg, h)
            h = h.flatten(1) if l < self.num_layers - 1 else h.mean(1)
        return h


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(
                block,
                h,
                edge_weight=block.edata["edge_weights"]
                if "edge_weights" in block.edata
                else None,
            )
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, use_uva, num_workers):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata["h"] = g.ndata["features"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["h"]
        )
        pin_memory = g.device != device and use_uva
        dataloader = dgl.dataloading.DataLoader(
            g,
            th.arange(g.num_nodes(), dtype=g.idtype, device=g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=use_uva,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        self.eval()

        for l, layer in enumerate(self.layers):
            y = th.empty(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                dtype=g.ndata["h"].dtype,
                device=g.device,
                pin_memory=pin_memory,
            )
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)
                if l < len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0].item() : output_nodes[-1].item() + 1] = h.to(
                    y.device
                )
            g.ndata["h"] = y
        return y
