import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from gcn_ns_sc import GCNSampling, GCNInfer


def copy_from_kvstore(nf, g, ndata_names):
    num_bytes = 0
    start = time.time()
    for i in range(nf.num_layers):
        for name in ndata_names:
            data = g.get_ndata(name, nf.layer_parent_nid(i))
            nf._node_frames[i][name] = data
            num_bytes += np.prod(data.shape)
    return num_bytes * 4, time.time() - start

def gcn_ns_train(g, args, cuda, n_classes, train_nid, test_nid):
    in_feats = args.n_features

    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        F.relu,
                        args.dropout)

    if cuda:
        model.cuda()

    loss_fcn = nn.CrossEntropyLoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           F.relu)

    if cuda:
        infer_model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        for nf in dgl.contrib.sampling.NeighborSampler(g.g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            copy_from_kvstore(nf, g, ['feats', 'labels'])
            model.train()
            # forward
            pred = model(nf)
            batch_labels = nf.layers[-1].data['labels'].long()
            loss = loss_fcn(pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for infer_param, param in zip(infer_model.parameters(), model.parameters()):
            infer_param.data.copy_(param.data)

        num_acc = 0.

        for nf in dgl.contrib.sampling.NeighborSampler(g.g, args.test_batch_size,
                                                       100,
                                                       neighbor_type='in',
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid):
            copy_from_kvstore(nf, g, ['feats', 'labels'])
            infer_model.eval()
            with torch.no_grad():
                pred = infer_model(nf)
                batch_labels = nf.layers[-1].data['labels'].long()
                num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

        print("Test Accuracy {:.4f}". format(num_acc/len(test_nid)))
