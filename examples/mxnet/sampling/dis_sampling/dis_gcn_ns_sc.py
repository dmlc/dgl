import os, sys
import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
parentdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from gcn_ns_sc import NodeUpdate, GCNSampling, GCNInfer


def gcn_ns_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples):
    n0_feats = g.nodes[0].data['features']
    in_feats = n0_feats.shape[1]
    g_ctx = n0_feats.context

    degs = g.in_degrees().astype('float32').as_in_context(g_ctx)
    norm = mx.nd.expand_dims(1./degs, 1)
    g.set_n_repr({'norm': norm})

    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        mx.nd.relu,
                        args.dropout,
                        prefix='GCN')

    model.initialize(ctx=ctx)
    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           mx.nd.relu,
                           prefix='GCN')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create('local'))

    # Create sampler receiver
    sampler = dgl.contrib.sampling.SamplerReceiver(graph=g, addr=args.ip, num_sender=args.num_sampler)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        for nf in sampler:
            nf.copy_from_parent(ctx=ctx)
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.
        num_tests = 0

        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid):
            nf.copy_from_parent(ctx=ctx)
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1)
            batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
            num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
            num_tests += nf.layer_size(-1)
            break

        print("Test Accuracy {:.4f}". format(num_acc/num_tests))
