#
# Copyright (c) 2022 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# based on node_classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import time
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.contrib import DataLoader2, SampledGraphSource, TensorFeatureSource, \
    EdgeWiseGraphSource
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse


class InferenceLayerGraphSource:
    def __init__(self, graph):
        self._graph = graph
        # get a permutation to sort by destination
        _, self._perm = torch.sort(self._graph.edges(form='uv')[1])

    def fetch_graph(self, idx, output_device):
        idx = self._perm[idx]
        subgraph = self._graph.edge_subgraph(edges=idx, relabel_nodes=False,
            store_ids=True).to(output_device)
        block = dgl.to_block(subgraph, include_dst_in_src=True)
        return [block]



class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        graph_source = InferenceLayerGraphSource(g)
        dataloader = DataLoader2(
                graph_source, None, torch.arange(g.num_edges()).to(g.device),
                output_device=device, batch_size=batch_size, shuffle=False,
                drop_last=False)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for blocks in tqdm.tqdm(dataloader):
                input_nodes = blocks[0].srcdata[dgl.NID]
                output_nodes = blocks[-1].dstdata[dgl.NID]
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] += h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, blocks in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        return MF.accuracy(pred, label)

def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler([10, 10, 10])

    # uva features
    feat_source = TensorFeatureSource(
        input_feats={'feat': g.ndata['feat'].pin_memory()},
        output_feats={'label': g.ndata['label'].pin_memory()})

    # store graph on GPU - only csc and without features
    with g.local_scope():
        g_dev = g.formats(['csc'])
        g_dev.ndata.pop('feat')
        g_dev.ndata.pop('label')
        g_dev = g_dev.to(device)
    graph_source = SampledGraphSource(g_dev, sampler)

    train_dataloader = DataLoader2(graph_source, feat_source, train_idx,
                                   batch_size=1024, shuffle=True,
                                   drop_last=False, output_device=device,
                                   num_workers=0)

    val_dataloader = DataLoader2(graph_source, feat_source, val_idx,
                                 batch_size=1024, shuffle=True,
                                 drop_last=False, output_device=device,
                                 num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        start = time.time()
        model.train()
        total_loss = 0
        for it, blocks in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time {:.4f} | GPU {:.1f} MB"
            .format(epoch, total_loss / (it+1), acc.item(), (time.time()-start), gpu_mem_alloc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print(f'Training with graph in GPU and features in UVA.')

    # load and preprocess dataset
    print('Loading data')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    g = g.to('cpu')
    device = torch.device('cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print('Training...')
    train(args, device, g, dataset, model)

    # test the model
    print('Testing...')
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=131072)
    print("Test Accuracy {:.4f}".format(acc.item()))
