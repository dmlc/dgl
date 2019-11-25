import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class NodeUpdate(nn.Module):
    def __init__(self, layer_id, in_feats, out_feats, dropout, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        self.linear = nn.Linear(in_feats, out_feats)
        self.dropout = None
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            norm = node.data['norm']
            h = h * norm
        else:
            agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
            agg_history = node.data[agg_history_str]
            # control variate
            h = h + agg_history
            if self.dropout:
                h = self.dropout(h)
        h = self.linear(h)
        if self.concat:
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCNSampling, self).__init__()
        self.n_layers = n_layers
        self.dropout = None
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        # input layer
        self.linear = nn.Linear(in_feats, n_hidden)
        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(i, n_hidden, n_hidden, dropout, activation, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, dropout))

    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        if self.dropout:
            h = self.dropout(h)
        h = self.linear(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = torch.cat((h, self.activation(h)), dim=1)
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            new_history = h.clone().detach()
            history_str = 'h_{}'.format(i)
            history = nf.layers[i].data[history_str]
            h = h - history

            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                             layer)
            h = nf.layers[i+1].data.pop('activation')
            # update history
            if i < nf.num_layers-1:
                nf.layers[i].data[history_str] = new_history

        return h


class GCNInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCNInfer, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        # input layer
        self.linear = nn.Linear(in_feats, n_hidden)
        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(i, n_hidden, n_hidden, 0, activation, True, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(n_layers, 2*n_hidden, n_classes, 0, None, True))

    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        h = self.linear(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = torch.cat((h, self.activation(h)), dim=1)
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = nf.layers[i+1].data.pop('activation')

        return h


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    norm = 1. / g.in_degrees().float().unsqueeze(1)

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
        norm = norm.cuda()

    g.ndata['features'] = features

    num_neighbors = args.num_neighbors
    n_layers = args.n_layers

    g.ndata['norm'] = norm

    g.update_all(fn.copy_src(src='features', out='m'),
                 fn.sum(msg='m', out='preprocess'),
                 lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})

    for i in range(n_layers):
        g.ndata['h_{}'.format(i)] = torch.zeros(features.shape[0], args.n_hidden).to(device=features.device)

    g.ndata['h_{}'.format(n_layers-1)] = torch.zeros(features.shape[0], 2*args.n_hidden).to(device=features.device)


    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        n_layers,
                        F.relu,
                        args.dropout)

    loss_fcn = nn.CrossEntropyLoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           n_layers,
                           F.relu)

    if cuda:
        model.cuda()
        infer_model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    for epoch in range(args.n_epochs):
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=n_layers,
                                                       seed_nodes=train_nid):
            for i in range(n_layers):
                agg_history_str = 'agg_h_{}'.format(i)
                g.pull(nf.layer_parent_nid(i+1).long(), fn.copy_src(src='h_{}'.format(i), out='m'),
                       fn.sum(msg='m', out=agg_history_str),
                       lambda node : {agg_history_str: node.data[agg_history_str] * node.data['norm']})

            node_embed_names = [['preprocess', 'h_0']]
            for i in range(1, n_layers):
                node_embed_names.append(['h_{}'.format(i), 'agg_h_{}'.format(i-1)])
            node_embed_names.append(['agg_h_{}'.format(n_layers-1)])
            nf.copy_from_parent(node_embed_names=node_embed_names)

            model.train()
            # forward
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device).long()
            batch_labels = labels[batch_nids]
            loss = loss_fcn(pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            node_embed_names = [['h_{}'.format(i)] for i in range(n_layers)]
            node_embed_names.append([])
            nf.copy_to_parent(node_embed_names=node_embed_names)


        for infer_param, param in zip(infer_model.parameters(), model.parameters()):
            infer_param.data.copy_(param.data)


        num_acc = 0.

        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_workers=32,
                                                       num_hops=n_layers,
                                                       seed_nodes=test_nid):
            node_embed_names = [['preprocess']]
            for i in range(n_layers):
                node_embed_names.append(['norm'])
            nf.copy_from_parent(node_embed_names=node_embed_names)

            infer_model.eval()
            with torch.no_grad():
                pred = infer_model(nf)
                batch_nids = nf.layer_parent_nid(-1).to(device=pred.device).long()
                batch_labels = labels[batch_nids]
                num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

        print("Test Accuracy {:.4f}". format(num_acc/n_test_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=2,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)


