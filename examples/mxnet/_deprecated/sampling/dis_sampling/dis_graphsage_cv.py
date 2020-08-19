import os, sys
import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import argparse, time, math
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
parentdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from graphsage_cv import GraphSAGELayer, NodeUpdate
from graphsage_cv import GraphSAGETrain, GraphSAGEInfer

def graphsage_cv_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples, distributed):
    n0_feats = g.nodes[0].data['features']
    num_nodes = g.number_of_nodes()
    in_feats = n0_feats.shape[1]
    g_ctx = n0_feats.context

    norm = mx.nd.expand_dims(1./g.in_degrees().astype('float32'), 1)
    g.set_n_repr({'norm': norm.as_in_context(g_ctx)})
    degs = g.in_degrees().astype('float32').asnumpy()
    degs[degs > args.num_neighbors] = args.num_neighbors
    g.set_n_repr({'subg_norm': mx.nd.expand_dims(mx.nd.array(1./degs, ctx=g_ctx), 1)})
    n_layers = args.n_layers

    g.update_all(fn.copy_src(src='features', out='m'),
                 fn.sum(msg='m', out='preprocess'),
                 lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
    for i in range(n_layers):
        g.init_ndata('h_{}'.format(i), (num_nodes, args.n_hidden), 'float32')
        g.init_ndata('agg_h_{}'.format(i), (num_nodes, args.n_hidden), 'float32')

    model = GraphSAGETrain(in_feats,
                           args.n_hidden,
                           n_classes,
                           n_layers,
                           args.dropout,
                           prefix='GraphSAGE')

    model.initialize(ctx=ctx)

    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GraphSAGEInfer(in_feats,
                                 args.n_hidden,
                                 n_classes,
                                 n_layers,
                                 prefix='GraphSAGE')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    kv_type = 'dist_sync' if distributed else 'local'
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create(kv_type))

    # Create sampler receiver
    sampler = dgl.contrib.sampling.SamplerReceiver(graph=g, addr=args.ip, num_sender=args.num_sampler)

    # initialize graph
    dur = []

    adj = g.adjacency_matrix(transpose=False).as_in_context(g_ctx)
    for epoch in range(args.n_epochs):
        start = time.time()
        if distributed:
            msg_head = "Worker {:d}, epoch {:d}".format(g.worker_id, epoch)
        else:
            msg_head = "epoch {:d}".format(epoch)
        for nf in sampler:
            for i in range(n_layers):
                agg_history_str = 'agg_h_{}'.format(i)
                dests = nf.layer_parent_nid(i+1).as_in_context(g_ctx)
                # TODO we could use DGLGraph.pull to implement this, but the current
                # implementation of pull is very slow. Let's manually do it for now.
                agg = mx.nd.dot(mx.nd.take(adj, dests), g.nodes[:].data['h_{}'.format(i)])
                g.set_n_repr({agg_history_str: agg}, dests)

            node_embed_names = [['preprocess', 'features', 'h_0']]
            for i in range(1, n_layers):
                node_embed_names.append(['h_{}'.format(i), 'agg_h_{}'.format(i-1), 'subg_norm', 'norm'])
            node_embed_names.append(['agg_h_{}'.format(n_layers-1), 'subg_norm', 'norm'])

            nf.copy_from_parent(node_embed_names=node_embed_names, ctx=ctx)
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
                loss = loss_fcn(pred, batch_labels)
                if distributed:
                    loss = loss.sum() / (len(batch_nids) * g.num_workers)
                else:
                    loss = loss.sum() / (len(batch_nids))

            loss.backward()
            trainer.step(batch_size=1)

            node_embed_names = [['h_{}'.format(i)] for i in range(n_layers)]
            node_embed_names.append([])

            nf.copy_to_parent(node_embed_names=node_embed_names)
        mx.nd.waitall()
        print(msg_head + ': training takes ' + str(time.time() - start))

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.
        num_tests = 0

        if not distributed or g.worker_id == 0:
            for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                           g.number_of_nodes(),
                                                           neighbor_type='in',
                                                           num_hops=n_layers,
                                                           seed_nodes=test_nid,
                                                           add_self_loop=True):
                node_embed_names = [['preprocess', 'features']]
                for i in range(n_layers):
                    node_embed_names.append(['norm', 'subg_norm'])
                nf.copy_from_parent(node_embed_names=node_embed_names, ctx=ctx)

                pred = infer_model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
                num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
                num_tests += nf.layer_size(-1)
                if distributed:
                    g._sync_barrier()
                print(msg_head + ": Test Accuracy {:.4f}". format(num_acc/num_tests))
                break
        elif distributed:
                g._sync_barrier()
