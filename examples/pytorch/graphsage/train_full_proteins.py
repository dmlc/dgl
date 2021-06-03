"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import os, sys
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from dgl.data.utils import load_tensors, load_graphs
from load_graph import load_reddit, load_ogb
from scipy.io import mmread
import requests
import random
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors

def download_proteins():
    print("Downloading dataset...")
    print("This might a take while..")
    url = "https://portal.nersc.gov/project/m1982/GNN/subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx"
    r = requests.get(url)
    with open("proteins.mtx", "wb") as handle:
        handle.write(r.content)
    

def proteins_mtx2dgl():
    print("Converting mtx2dgl..")
    print("This might a take while..")
    #a = mmread('subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx')
    a = mmread('proteins.mtx')
    coo = a.tocoo()
    u = torch.tensor(coo.row, dtype=torch.int64)
    v = torch.tensor(coo.col, dtype=torch.int64)
    #print("Edges in the graph: ", u.shape[0])
    g = dgl.DGLGraph()
    g.add_edges(u,v)
    #g.add_edges(v,u)

    n = g.number_of_nodes()
    feat_size = 128
    feats = torch.empty([n, feat_size], dtype=torch.float32)

    train_size = 1000000
    test_size = 500000
    val_size = 5000
    nlabels = 256

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    label      = torch.zeros(n, dtype=torch.int64)

    for i in range(train_size):
        train_mask[i] = True

    for i in range(test_size):
        test_mask[train_size + i] = True

    for i in range(val_size):
        val_mask[train_size + test_size + i] = True

    for i in range(n):
        label[i] = random.choice(range(nlabels))

    g.ndata['feat'] = feats
    g.ndata['train_mask'] = train_mask
    g.ndata['test_mask'] = test_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['label'] = label

    print(g)
    return g


def save(g, dataset):
    print("Saving dataset..")
    part_dir = os.path.join("./" + dataset)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    os.makedirs(part_dir, mode=0o775, exist_ok=True)
    save_tensors(node_feat_file, g.ndata)
    save_graphs(part_graph_file, [g])
    #print("Graph saved successfully !!")
    

def load_proteins(dataset):    
    part_dir = dataset
    #node_feats_file = os.path.join(part_dir + "/node_feat.dgl")
    graph_file = os.path.join(part_dir + "/graph.dgl")

    if not os.path.exists("proteins.mtx"):
        download_proteins()
    if not os.path.exists(graph_file):        
        g = proteins_mtx2dgl()
        save(g, dataset)
    
    #node_feats = load_tensors(node_feats_file)
    graph = load_graphs(graph_file)[0][0]
    #graph.ndata = node_feats
    
    return graph


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        #h = self.dropout(inputs)
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)



def main(args):
    # load and preprocess dataset
    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        g, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        g, _ = load_ogb('ogbn-papers100M')
    elif args.dataset == 'proteins':
        g = load_proteins('proteins')        
    else:    
        data = load_data(args)
        g = data[0]
        
    features = g.ndata['feat']
    try:
        labels = g.ndata['label']
    except:
        labels = g.ndata['labels']
        
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    #n_classes = data.num_classes
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Node %d
      #Edges %d
      #Features %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (g.number_of_nodes(),
           n_edges, in_feats, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    use_gpu = False
    enable_profiling = False
    with torch.autograd.profiler.profile(enable_profiling, use_gpu, True) as prof:    
      for epoch in range(args.n_epochs):
        tic = time.time()
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        toc = time.time()
        #acc = evaluate(model, g, features, labels, val_nid)
        #print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                    acc, n_edges / np.mean(dur) / 1000))
        print("Epoch: {:}, time: {:0.4f} sec".format(epoch, toc - tic))

    if enable_profiling:
        with open("ogb.prof", "w") as prof_f:
            prof_f.write(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time_total"))
        
    print()
    acc = evaluate(model, g, features, labels, test_nid)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.004,  ##0.03
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, ##0
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)
