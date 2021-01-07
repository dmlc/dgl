import dgl.function as fn
def GRANDConv(graph, feats, order):
    
    with graph.local_scope():
        
        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        x = feats
        y = 0+feats

        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return x


import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from model import GRAND
from utils import consis_loss

import warnings
warnings.filterwarnings('ignore')

def argument():

    parser = argparse.ArgumentParser(description='GRAND')

    # data source params
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 reg.')
    # model params
    parser.add_argument("--hid_dim", type=int, default=32, help='Hidden layer dimensionalities.')
    parser.add_argument('--dropnode_rate', type=float, default=0.5,
                        help='Dropnode rate (1 - keep probability).')
    parser.add_argument('--input_droprate', type=float, default=0.0,
                    help='dropout rate of input layer')
    parser.add_argument('--hidden_droprate', type=float, default=0.0,
                    help='dropout rate of hidden layer')
    parser.add_argument('--order', type=int, default=8, help='Propagation step')
    parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
    parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=1., help='Lamda')
    parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')

    args = parser.parse_args()
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:%s'%args.gpu
    else:
        args.device = 'cpu'

    return args

if __name__ == '__main__':

    # Step 1： Prepare graph data and retrieve train/validation/test index  #
    args = argument()
    print(args)
    if args.dataname == 'cora':
        dataset = CoraGraphDataset()
    elif args.dataname == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataname == 'pubmed':
        dataset = PubmedGraphDataset()
        
    graph = dataset[0]
    
    graph = dgl.add_self_loop(graph)
    device = args.device

    n_classes = dataset.num_classes

    labels = graph.ndata.pop('label').to(device).long()
    feats = graph.ndata.pop('feat').to(device)
    
    n_features = feats.shape[-1]

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    
    in_dim = n_features
    hid_dim = args.hid_dim
    n_class = n_classes
    K = args.order
    S = args.sample
    T = args.tem
    lam = args.lam
    batchnorm = args.use_bn
    
    node_dropout = args.dropnode_rate
    input_droprate = args.input_droprate
    hidden_droprate = args.hidden_droprate

    model = GRAND(in_dim, hid_dim, n_class, S, K,
                  node_dropout, input_droprate, 
                  hidden_droprate, batchnorm)

    
    model = model.to(device)
    graph = graph.to(device)
    

    loss_fn = nn.NLLLoss()
    opt = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    best_test_acc = 0
    best_val_acc = 0
    
    loss_best = np.inf
    acc_best = 0

    val_loss_list = []
    val_acc_list = []
    
    loss_min = np.inf
    acc_max = 0.0
    
    for epoch in range(args.epochs):

        
        ''' Training '''
        model.train()
        
        loss_sup = 0
        logits = model(graph, feats, True)
        
        K = len(logits)
        for k in range(K):
            loss_sup += F.nll_loss(logits[k][train_idx], labels[train_idx])
        
        loss_sup = loss_sup/K
        loss_consis = consis_loss(logits, args.tem, args.lam)
        
        loss_train = loss_sup + loss_consis
        acc_train = th.sum(logits[0][train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

        opt.zero_grad()
        loss_train.backward()
        opt.step()

        ''' Validation '''
        model.eval()
        with th.no_grad():
        
            val_logits = model(graph, feats, False)
            
            loss_val = F.nll_loss(val_logits[val_idx], labels[val_idx]) 
            acc_val = th.sum(val_logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

            print("In epoch {}, Train Acc: {:.4f} | Train Loss: {:.4f} ,Val Acc: {:.4f} | Val Loss: {:.4f}".
              format(epoch, acc_train, loss_train.item(), acc_val, loss_val.item()))

            if loss_val < loss_best or acc_val > acc_best:
                if loss_val < loss_best:
                    best_epoch = epoch
                    th.save(model.state_dict(), args.dataname +'.pkl')
                no_improvement = 0
                loss_best = min(loss_val, loss_best)
                acc_best = max(acc_val, acc_best)
            else:
                no_improvement += 1
                if no_improvement == args.early_stopping:
                    print('Early stopping.')
                    break
        
    print("Optimization Finished!")
    
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(th.load(args.dataname +'.pkl'))
    
    ''' Testing '''
    model.eval()
    
    test_logits = model(graph, feats, False)  
    test_acc = th.sum(test_logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f}".format(test_acc))

