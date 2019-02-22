"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

class SGCLayer(nn.Module):
    def __init__(self,
                 g,
                 h,
                 in_feats,
                 out_feats,
                 bias=False,
                 K=2):
        super(SGCLayer, self).__init__()
        self.g = g
        self.weight = nn.Linear(in_feats, out_feats, bias=bias)
        self.K = K
        # precomputing message passing
        for _ in range(self.K):
            # normalization by square root of src degree
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
        # store precomputed result into a cached variable
        self.cached_h = h

    def forward(self, mask):
        h = self.weight(self.cached_h[mask])
        return h

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(mask) # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

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

    # graph preprocess and calculate normalization factor
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create SGC model
    model = SGCLayer(g,
                     features,
                     in_feats,
                     n_classes,
                     args.bias,
                     K=2)

    if cuda: model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(train_mask) # only compute the train set
        loss = loss_fcn(logits, labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.2,
            help="learning rate")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-6,
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
