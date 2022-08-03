import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import time
import random
import tqdm
import argparse
from dgl.utils.dgi.infer_helper import InferenceHelper
import dgl.backend as backend

def load_reddit():
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=True)
    g = data[0]
    return g, data.num_classes

def load_data(args):
    dataset = load_reddit()
    dim = 100
    return dataset, dim

def train(args):
    dataset, dim = load_data(args)
    g = dataset[0]

    feat = np.random.rand(g.number_of_nodes(), dim)
    feat = backend.tensor(feat, dtype=backend.data_type_dict['float32'])

    train_mask = g.ndata['train_mask']
    labels = g.ndata['label']
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 25, 50])
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid, sampler,
        batch_size=2000,
        shuffle=True,
        drop_last=False,
        num_workers=4)

    num_classes = dataset[1]
    in_feats = feat.shape[1]
    hidden_feature = args.num_hidden
    if args.model == "GCN":
        model = StochasticTwoLayerGCN(args.num_layers, in_feats, hidden_feature, num_classes)
    else:
        raise NotImplementedError()

    if args.gpu == -1:
        device = "cpu"
    else:
        device = "cuda:" + str(args.gpu)
    model = model.to(torch.device(device))
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device(device)) for b in blocks]
            input_features = feat[input_nodes].to(torch.device(device))
            pred = model(blocks, input_features)
            output_labels = labels[output_nodes].to(torch.device(device))
            loss = loss_fcn(pred, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # We do not need to train the network, just to make sure it can run.
            break

    with torch.no_grad():
        helper = InferenceHelper(model, (), 2000, torch.device(device), debug = True)
        torch.cuda.synchronize()
        st = time.time()
        helper_pred = helper.inference(g, feat)
        cost_time = time.time() - st
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
        print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ssd', help="use ssd", action="store_true")
    argparser.add_argument('--reorder', help="use the reordered graph", action="store_true")
    argparser.add_argument('--use-uva', help="use the pinned memory", action="store_true")
    argparser.add_argument('--free-rate', help="free memory rate", type=float, default=0.9)

    # Different inference mode. 
    argparser.add_argument('--topdown', action="store_true")
    argparser.add_argument('--cpufull', action="store_true")
    argparser.add_argument('--gpufull', action="store_true")
    argparser.add_argument('--gpu', help="GPU device ID. Use -1 for CPU training", type=int, default=0)
    argparser.add_argument('--auto', action="store_true")

    argparser.add_argument('--model', help="can be GCN, GAT, SAGE and JKNET", type=str, default='GCN')
    argparser.add_argument('--debug', action="store_true")
    argparser.add_argument('--num-epochs', type=int, default=0)
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-heads', type=int, default=2)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=2000)
    argparser.add_argument('--load-data', action="store_true")
    args = argparser.parse_args()

    if args.load_data:
        g, n_classes = load_data(args)
        print(g)
        
    else:
        train(args)
