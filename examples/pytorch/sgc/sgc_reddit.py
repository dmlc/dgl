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
    def __init__(self,g,h,in_feats,out_feats,K=2):
        super(SGCLayer, self).__init__()
        self.g = g
        self.weight = nn.Linear(in_feats, out_feats, bias=True)
        self.K = K
        # precomputing message passing
        start = time.perf_counter()
        for _ in range(self.K):
            # normalization by square root of src degree
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
        h = (h-h.mean(0))/h.std(0)
        precompute_elapse = time.perf_counter()-start
        print("Precompute Time(s): {:.4f}".format(precompute_elapse))
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
    args.dataset = "reddit-self-loop"
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
    start = time.perf_counter()
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda: norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    preprocess_elapse = time.perf_counter()-start
    print("Preprocessing Time: {:.4f}".format(preprocess_elapse))

    # create SGC model
    model = SGCLayer(g,features,in_feats,n_classes,K=2)

    if cuda: model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.LBFGS(model.parameters())

    # define loss closure
    def closure():
        optimizer.zero_grad()
        output = model(train_mask)
        loss_train = F.cross_entropy(output, labels[train_mask])
        loss_train.backward()
        return loss_train

    # initialize graph
    dur = []
    start = time.perf_counter()
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(train_mask) # only compute the train set
        loss = optimizer.step(closure)

    train_elapse = time.perf_counter()-start
    print("Train epoch {} | Train Time(s) {:.4f}".format(epoch, train_elapse))
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
    parser.add_argument("--n-epochs", type=int, default=2,
            help="number of training epochs")
    args = parser.parse_args()
    print(args)

    main(args)
