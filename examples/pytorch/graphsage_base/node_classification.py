import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, out_feats, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.out_feats = out_feats

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
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
        buffer_device = 'cpu'

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.out_feats,
                device=buffer_device, pin_memory=True)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # use an explicitly contiguous slice
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous so we can take
                # advantage of that here
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
    
def evaluate_batches(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))

def evaluate(model, graph, device, nid, batch_size):
    model.eval()
    nid = nid.to(device)
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid].to(device)
        label = graph.ndata['label'][nid].to(device)
        return MF.accuracy(pred, label)

def train(args, g, device, data):
    in_feats, n_classes, train_idx, valid_idx = data
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)

    # create GraphSAGE model
    model = SAGE(in_feats, n_hidden=256, out_feats=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sampler = dgl.dataloading.NeighborSampler([5, 10, 15],
                                              prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(g, train_idx, sampler, device = device,
                                                  batch_size=1024, shuffle=True,
                                                  drop_last=False, num_workers=0,
                                                  use_uva=True)

    valid_dataloader = dgl.dataloading.DataLoader(g, valid_idx, sampler, device=device,
                                                  batch_size=1024, shuffle=True,
                                                  drop_last=False, num_workers=0,
                                                  use_uva=True)
    
    for epoch in range(10):
        model.train()
        total_loss = torch.tensor(0.0).to(device)
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        acc = evaluate_batches(model, g, valid_dataloader)
        mem = torch.cuda.max_memory_allocated() / 1000000
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | "
              "GPU Mem(MB) {:.2f}".format(epoch, total_loss.item() / (it+1),
                                          acc.item(), mem))
    print()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="training device (0: gpu, -1: cpu)")
    args = parser.parse_args()
    # load and preprocess dataset
    dataset = DglNodePropPredDataset('ogbn-products')
    g, labels = dataset[0]
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g.ndata['label'] = labels.squeeze().to(g.device)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    # data logging 
    n_classes = dataset.num_classes
    n_edges = g.num_edges()
    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
    (n_edges, n_classes,
     len(train_idx), len(valid_idx), len(test_idx)))
    # model training
    in_feats = g.ndata['feat'].shape[1]
    data = in_feats, n_classes, train_idx, valid_idx
    model = train(args, g, device, data)
    # test accuracy and offline inference of all nodes                           
    acc = evaluate(model, g, device, test_idx, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
    
