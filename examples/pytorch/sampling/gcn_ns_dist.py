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
    first_layer_nid = torch.cat(first_layer_nid, dim=0)
    last_layer_nid = torch.cat(last_layer_nid, dim=0)

    # TODO we need to gracefully handle the case that the nodes don't exist.
    start = time.time()
    first_layer_data = g.get_ndata('features', first_layer_nid)
    last_layer_data = g.get_ndata('labels', last_layer_nid)
    stats[2] = time.time() - start
    first_layer_local = g.is_local(first_layer_nid).numpy()
    last_layer_local = g.is_local(last_layer_nid).numpy()
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
        print('epoch', epoch)
        start = time.time()
        num_bytes = 0
        num_local_bytes = 0
        copy_time = 0
        for_back_time = 0
        agg_grad_time = 0

        stats = [0, 0, 0]
        for nf in dgl.contrib.sampling.NeighborSampler(g.g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid,
                                                       prefetch=True,
                                                       prepare=lambda nfs: copy_from_kvstore(nfs, g, stats)):
            #copy_from_kvstore(nf, g, stats)
            nbytes, local_nbytes, copy_time1 = stats
            num_bytes += nbytes
            num_local_bytes += local_nbytes
            copy_time += copy_time1
            model.train()
            # forward
            start1 = time.time()
            pred = model(nf)
            batch_labels = nf.layers[-1].data['labels'].long()
            loss = loss_fcn(pred, batch_labels)
            loss.backward()
            for_back_time += (time.time() - start1)

            start1 = time.time()
            optimizer.zero_grad()
            optimizer.step()
            agg_grad_time += (time.time() - start1)

        for infer_param, param in zip(infer_model.parameters(), model.parameters()):
            infer_param.data.copy_(param.data)

        train_time = time.time() - start
        print('Trainer {}: Train Time {:.4f}, Throughput: {:.4f} MB/s, local throughput: {:.4f} MB/s'.format(
            g.get_id(), train_time, num_bytes / copy_time / 1024 / 1024, num_local_bytes / copy_time / 1024 / 1024),
            flush=True)
        print('Trainer {}: copy {:.4f}, forward_backward: {:.4f}, gradient update:{:.4f}'.format(
            g.get_id(), copy_time, for_back_time, agg_grad_time))

        num_acc = 0.

        for nf in dgl.contrib.sampling.NeighborSampler(g.g, args.test_batch_size,
                                                       100,
                                                       neighbor_type='in',
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid,
                                                       prefetch=True,
                                                       prepare=lambda nfs: copy_from_kvstore(nfs, g, stats)):
            #copy_from_kvstore(nf, g, stats)
            infer_model.eval()
            with torch.no_grad():
                pred = infer_model(nf)
                batch_labels = nf.layers[-1].data['labels'].long()
                num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

        print("Test Accuracy {:.4f}". format(num_acc/len(test_nid)))
