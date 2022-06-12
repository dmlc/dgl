import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, JumpingKnowledge
import tqdm
import numpy as np
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows
from dgi.utils import update_out_in_chunks

class Concate(nn.Module):
    def forward(self, g, jumped):
        g.srcdata['h'] = jumped
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return g.dstdata['h']

class JKNet(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=1,
                 mode='cat',
                 dropout=0.):
        super(JKNet, self).__init__()
        
        self.n_hidden = hid_dim
        self.n_classes = out_dim
        self.out_features = out_dim
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        if self.mode == 'lstm':
            self.jump = JumpingKnowledge(mode, hid_dim, num_layers)
        else:
            self.jump = JumpingKnowledge(mode)

        if self.mode == 'cat':
            hid_dim = hid_dim * (num_layers)

        self.output = nn.Linear(hid_dim, out_dim)
        self.agge = Concate()
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, g, feats):
        feat_lst = []
        for i, layer in enumerate(self.layers):
            feats = self.dropout(layer(g[i], feats))
            feat_lst.append(feats[:g[-1].num_src_nodes()])
        jumped = self.jump(feat_lst)
        agged = self.agge(g[-1], jumped)

        return self.output(agged)

    def forward_full(self, g, feats):
        feat_lst = []
        for layer in self.layers:
            feats = self.dropout(layer(g, feats))
            feat_lst.append(feats)

        jumped = self.jump(feat_lst)
        agged = self.agge(g, jumped)

        return self.output(agged)

    def inference(self, g, batch_size, device, x, nids, use_uva = False, use_ssd = False):
        for k in list(g.ndata.keys()):
            g.ndata.pop(k)
        for k in list(g.edata.keys()):
            g.edata.pop(k)

        feat_lst = []
        for l, layer in enumerate(self.layers):
            
            shape = (g.num_nodes(), self.n_hidden)
            if use_ssd:
                y = torch.as_tensor(np.memmap(f"/ssd/feat_{l}.npy",dtype=np.float32, mode="w+", shape=shape, ))
            else:
                y = torch.zeros(*shape)
            feat_lst.append(y)

            if use_uva:
                pin_memory_inplace(x)
                nids.to(device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                nids,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                use_uva=use_uva,
                drop_last=False,
                num_workers=0)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                torch.cuda.empty_cache()
                block = blocks[0]

                block = block.int().to(device)
                if use_uva:
                    h = gather_pinned_tensor_rows(x, input_nodes)
                else:
                    h = x[input_nodes].to(device)
                
                h = layer(block, h)
                h = self.dropout(h)

                update_out_in_chunks(feat_lst[-1], output_nodes, h)

                torch.cuda.reset_peak_memory_stats()

                torch.cuda.empty_cache()
            if use_uva:
                unpin_memory_inplace(x)
            x = feat_lst[-1]

        shape = (g.num_nodes(), self.n_classes)
        if use_ssd:
            y = torch.as_tensor(np.memmap(f"/ssd/feat_99.npy",dtype=np.float32, mode="w+", shape=shape, ))
        else:
            y = torch.zeros(*shape)

        if use_uva:
            for feat in feat_lst:
                pin_memory_inplace(feat)
            nids.to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            nids,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            use_uva=use_uva,
            drop_last=False,
            num_workers=0)

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            torch.cuda.empty_cache()
            block = blocks[0]

            block = block.int().to(device)
            h_lst = []
            for feat in feat_lst:
                if use_uva:
                    h_lst.append(gather_pinned_tensor_rows(feat, input_nodes))
                else:
                    h_lst.append(feat[input_nodes].to(device))

            jumped = self.jump(h_lst)
            agged = self.agge(block, jumped)
            output = self.output(agged)
            
            update_out_in_chunks(y, output_nodes, output)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        if use_uva:
            for feat in feat_lst:
                unpin_memory_inplace(feat)
        return y
