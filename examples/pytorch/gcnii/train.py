import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gcnii import GCNII

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        loss = F.cross_entropy(logits, labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return loss, correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
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
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCNII(in_size=in_feats,
                  out_size=n_classes,
                  hidden_size=args.n_hidden,
                  num_layers=args.n_layers,
                  alpha=args.alpha,
                  lamda=args.lamda,
                  dropout=args.dropout,
                  norm=True,
                  weight=True,
                  bias=False,
                  activation=F.relu)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam([
                        {'params':model.params_conv,'weight_decay':args.wd1},
                        {'params':model.params_dense,'weight_decay':args.wd2},],
                        lr=args.lr)

    # initialize graph
    dur = []

    best_val_loss = 999999999
    bad_counter = 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, g, features, labels, val_mask)
        if epoch % 10 == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(), val_loss, 
                                                 val_acc, n_edges / np.mean(dur) / 1000))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss, test_acc = evaluate(model, g, features, labels, test_mask)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= args.patience:
            break


    print()
    print("Test loss {:.4f} | Test accuracy {:.2%}".format(test_loss, test_acc))
    return(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCNII')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.6,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=1500,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=64,
            help="number of hidden gcn layers")
    parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on conv layer parameters).')
    parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on dense layer parameters).')
    parser.add_argument('--patience', type=int, default=100, help='patience')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.set_defaults(self_loop=True)
    args = parser.parse_args()
    print(args)

    main(args)
