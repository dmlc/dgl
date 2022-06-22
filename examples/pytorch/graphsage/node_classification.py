import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
from dgl.data import register_data_args, load_data
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import argparse
from model import GraphSAGEBatch

def batch_evaluate(model, graph, dataloader):
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

def train(args, g, device, in_feats, n_classes, train_idx, valid_idx):
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)

    model = GraphSAGEBatch(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sampler = dgl.dataloading.NeighborSampler([15, 10, 5],
                                              prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(g, train_idx, sampler, device = device,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  drop_last=False, num_workers=0,
                                                  use_uva=not args.pure_gpu)

    valid_dataloader = dgl.dataloading.DataLoader(g, valid_idx, sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  drop_last=False, num_workers=0,
                                                  use_uva=not args.pure_gpu)
    
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
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
        tt = time.time()
        dur.append(tt-t0)
        acc = batch_evaluate(model, g, valid_dataloader)
        mem = torch.cuda.max_memory_allocated() / 1000000
        print("Epoch {:05d} | Avg. Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "GPU Mem(MB) {:.2f}".format(epoch, np.mean(dur), total_loss.item() / (it+1),
                                            acc.item(), mem))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--pure-gpu', action='store_true',
                    help='Perform both sampling and training on GPU.')
    args = parser.parse_args()
    print(args)
    dataset = DglNodePropPredDataset('ogbn-products')
    g, labels = dataset[0]
    g.ndata['label'] = labels.squeeze()
    features = g.ndata['feat']
    split_idx = dataset.get_idx_split()
    in_feats = features.shape[1]
    n_classes = dataset.num_classes
    n_edges = g.number_of_edges()
    if args.gpu < 0:
        device = torch.device('cpu')
        args.pure_gpu = False
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = g.to(device if args.pure_gpu else 'cpu')
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
    (n_edges, n_classes,
     len(train_idx), len(valid_idx), len(test_idx)))
     
    model = train(args, g, device, in_feats, n_classes, train_idx, valid_idx)
    # Test accuracy and offline inference of all nodes                           
    acc = evaluate(model, g, device, test_idx, 4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
    
