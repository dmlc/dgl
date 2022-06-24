import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
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
        if isinstance(model, GraphSAGEBatch):
            pred = model.inference(graph, device, batch_size)
        else:
            pred = model.module.inference(graph, device, batch_size)
        pred = pred[nid].to(device)
        label = graph.ndata['label'][nid].to(device)
        return MF.accuracy(pred, label)

def train(args, g, device, data):
    in_feats, n_classes, train_idx, valid_idx = data
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)

    # create GraphSAGE model
    # parameters can be tuned here as keyword arguments
    model = GraphSAGEBatch(in_feats, n_classes, n_hidden=256, n_layers=1, aggregator_type = 'mean').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sampler = dgl.dataloading.NeighborSampler([5, 10, 15],
                                              prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(g, train_idx, sampler, device = device,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  drop_last=False, num_workers=0,
                                                  use_uva=not args.pure_gpu)

    valid_dataloader = dgl.dataloading.DataLoader(g, valid_idx, sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  drop_last=False, num_workers=0,
                                                  use_uva=not args.pure_gpu)
    
    time_sta = time.time()
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
        dur = tt-t0
        acc = batch_evaluate(model, g, valid_dataloader)
        mem = torch.cuda.max_memory_allocated() / 1000000
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "GPU Mem(MB) {:.2f}".format(epoch, dur, total_loss.item() / (it+1),
                                            acc.item(), mem))
    print("Training time(s) {:.4f}". format(time.time() - time_sta))
    print()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="mini-batch sample size")
    # need to switch to --graph-device, in line with multi-gpu-node-classification
    parser.add_argument('--pure-gpu', action='store_true',
                    help='Perform both sampling and training on GPU.')
    args = parser.parse_args()
    print(args)
    
    # load and preprocess dataset
    dataset = DglNodePropPredDataset('ogbn-products')
    g, labels = dataset[0]
    if args.gpu < 0:
        device = torch.device('cpu')
        args.pure_gpu = False
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = g.to(device if args.pure_gpu else 'cpu')
        
    g.ndata['label'] = labels.squeeze().to(g.device)
    features = g.ndata['feat']
    split_idx = dataset.get_idx_split()
    in_feats = features.shape[1]
    n_classes = dataset.num_classes
    n_edges = g.num_edges()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
    (n_edges, n_classes,
     len(train_idx), len(valid_idx), len(test_idx)))

    data = in_feats, n_classes, train_idx, valid_idx
    model = train(args, g, device, data)
    # Test accuracy and offline inference of all nodes                           
    acc = evaluate(model, g, device, test_idx, 4096) # 4096: batch size for evaluation
    print("Test Accuracy {:.4f}".format(acc.item()))
    
