import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import tqdm

def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            return g.dstdata['y']


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(GraphConv(n_hidden, n_hidden))
        self.convs.append(GraphConv(n_hidden, n_classes))

    def forward(self, blocks, x):
        for i, (conv, block) in enumerate(zip(self.convs, self.blocks)):
            x = conv(block, x, block.edata['w'])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

    def inference(self, g, x, w, batch_size, device, num_workers):
        for l, layer in enumerate(self.convs):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)

                y[output_nodes] = h.cpu()

            x = y
        return y
