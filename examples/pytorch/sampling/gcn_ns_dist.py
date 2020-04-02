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

def copy_from_kvstore(nfs, g, stats):
    num_bytes = 0
    num_local_bytes = 0
    first_layer_nid = []
    last_layer_nid = []
    first_layer_offs = [0]
    last_layer_offs = [0]
    for nf in nfs:
        first_layer_nid.append(nf.layer_parent_nid(0))
        last_layer_nid.append(nf.layer_parent_nid(-1))
        first_layer_offs.append(first_layer_offs[-1] + len(nf.layer_parent_nid(0)))
        last_layer_offs.append(last_layer_offs[-1] + len(nf.layer_parent_nid(-1)))
    first_layer_nid = mx.nd.concat(*first_layer_nid, dim=0)
    last_layer_nid = mx.nd.concat(*last_layer_nid, dim=0)

    # TODO we need to gracefully handle the case that the nodes don't exist.
    start = time.time()
    first_layer_data = g.get_ndata('features', first_layer_nid)
    last_layer_data = g.get_ndata('labels', last_layer_nid)
    stats[2] = time.time() - start
    first_layer_local = g.is_local(first_layer_nid).asnumpy()
    last_layer_local = g.is_local(last_layer_nid).asnumpy()
    num_bytes += np.prod(first_layer_data.shape)
    num_bytes += np.prod(last_layer_data.shape)
    if len(first_layer_data.shape) == 1:
        num_local_bytes += np.sum(first_layer_local)
    else:
        num_local_bytes += np.sum(first_layer_local) * first_layer_data.shape[1]
    if len(last_layer_data.shape) == 1:
        num_local_bytes += np.sum(last_layer_local)
    else:
        num_local_bytes += np.sum(last_layer_local) * last_layer_data.shape[1]

    for idx, nf in enumerate(nfs):
        start = first_layer_offs[idx]
        end = first_layer_offs[idx + 1]
        nfs[idx]._node_frames[0]['features'] = first_layer_data[start:end]
        start = last_layer_offs[idx]
        end = last_layer_offs[idx + 1]
        nfs[idx]._node_frames[-1]['labels'] = last_layer_data[start:end]

    stats[0] = num_bytes * 4
    stats[1] = num_local_bytes * 4

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
        stats = [0, 0, 0]
        for nf in dgl.contrib.sampling.NeighborSampler(g.g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            copy_from_kvstore(nf, g, stats)
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
            copy_from_kvstore(nf, g, stats)
            infer_model.eval()
            with torch.no_grad():
                pred = infer_model(nf)
                batch_labels = nf.layers[-1].data['labels'].long()
                num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

        print("Test Accuracy {:.4f}". format(num_acc/len(test_nid)))
