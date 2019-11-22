"""Training script"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import mxnet as mx
from mxnet import gluon
from data import MovieLens
from model_es import GCMCLayer, BiDecoder
from utils import get_activation, parse_ctx, gluon_net_info, gluon_total_param_num, \
                  params_clip_global_norm, MetricLogger
from mxnet.gluon import Block

import dgl
import gc

class Net(Block):
    def __init__(self, args, **kwargs):
        super(Net, self).__init__(**kwargs)
        self._act = get_activation(args.model_activation)
        self._ctx = args.ctx
        with self.name_scope():
            self.encoder = GCMCLayer(args.rating_vals,
                                     args.src_in_units,
                                     args.dst_in_units,
                                     args.gcn_agg_units,
                                     args.gcn_out_units,
                                     args.gcn_dropout,
                                     args.gcn_agg_accum,
                                     agg_act=self._act,
                                     share_user_item_param=args.share_param)
            self.decoder = BiDecoder(args.rating_vals,
                                     in_units=args.gcn_out_units,
                                     num_basis_functions=args.gen_r_num_basis_func)

    def forward(self, head_subgraph, tail_subgraph, true_head_idx, true_tail_idx):
        user_out, movie_out = self.encoder(
            head_subgraph,
            tail_subgraph,
            self._ctx)

        true_user_out = user_out[true_head_idx]
        true_movie_out = movie_out[true_tail_idx]

        pred_ratings = self.decoder(true_user_out, true_movie_out)
        return pred_ratings

def evaluate(args, net, dataset, segment='valid'):
    print('eval now')
    g_user_fea = mx.nd.zeros((dataset.num_user))
    g_movie_fea = mx.nd.zeros((dataset.num_movie))
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    num_edges = rating_values.shape[0]
    edges = mx.nd.arange(num_edges, dtype='int64')
    real_pred_ratings = []

    #for sample_idx in range(0, (num_edges + 1000 - 1) // 1000):
    if True:
        edge_ids = edges #edges[sample_idx * 1000: (sample_idx + 1) * 1000 if (sample_idx + 1) * 1000 < num_edges else num_edges]
        head_ids, tail_ids = dec_graph.find_edges(edge_ids)

        head_subgraphs = {}
        tail_subgraphs = {}
        for i, rating in enumerate(args.rating_vals):
            t = enc_graph.canonical_etypes[i * 2]
            rev_t = enc_graph.canonical_etypes[i * 2 + 1]

            head_in_edges = enc_graph.in_edges(head_ids, 'eid', etype=rev_t)
            tail_in_edges = enc_graph.in_edges(tail_ids, 'eid', etype=t)

            head_subgraphs[rev_t] = head_in_edges
            tail_subgraphs[t] = tail_in_edges

        head_subgraph = enc_graph.edge_subgraph(head_subgraphs)
        tail_subgraph = enc_graph.edge_subgraph(tail_subgraphs)

        head_NID = head_subgraph.nodes['user'].data[dgl.NID]
        tail_NID = tail_subgraph.nodes['movie'].data[dgl.NID]

        g_user_fea[head_NID] = mx.nd.arange(head_NID.shape[0], dtype='int32')
        g_movie_fea[tail_NID] = mx.nd.arange(tail_NID.shape[0], dtype='int32')

        true_head_idx = g_user_fea[head_ids].as_in_context(args.ctx)
        true_tail_idx = g_movie_fea[tail_ids].as_in_context(args.ctx)

        # Evaluate RMSE
        with mx.autograd.predict_mode():
            pred_ratings = net(head_subgraph, tail_subgraph,
                               true_head_idx, true_tail_idx)
        real_pred_rating = (mx.nd.softmax(pred_ratings, axis=1) *
                            nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
        real_pred_ratings.append(real_pred_rating)

    real_pred_ratings = mx.nd.concat(*real_pred_ratings, dim=0)
    rmse = mx.nd.square(real_pred_ratings - rating_values).mean().asscalar()
    rmse = np.sqrt(rmse)
    return rmse

def train(args):
    print(args)
    dataset = MovieLens(args.data_name, args.ctx, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    ### build the net
    net = Net(args=args)
    net.initialize(init=mx.init.Xavier(factor_type='in'), ctx=args.ctx)
    net.hybridize()
    nd_possible_rating_values = mx.nd.array(dataset.possible_rating_values, ctx=args.ctx, dtype=np.float32)
    rating_loss_net = gluon.loss.SoftmaxCELoss()
    rating_loss_net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), args.train_optimizer, {'learning_rate': args.train_lr})
    print("Loading network finished ...\n")

    ### perpare training data
    train_gt_labels = dataset.train_labels
    train_gt_ratings = dataset.train_truths

    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'idx', 'loss', 'rmse'], ['%d', '%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    avg_gnorm = 0
    count_rmse = 0
    count_num = 0
    count_loss = 0
    total_nodes = dataset.num_user + dataset.num_movie

    enc_graph = dataset.train_enc_graph
    g_user_fea = mx.nd.zeros((dataset.num_user,))
    g_movie_fea = mx.nd.zeros((dataset.num_movie,))
    train_truths = dataset.train_truths
    train_labels = dataset.train_labels

    print("Start training ...")
    dur = []
    for iter_idx in range(1, args.train_max_iter):
        if iter_idx > 3:
            t0 = time.time()

        num_edges = dataset.train_truths.shape[0]
        seed = mx.nd.arange(num_edges, dtype='int64')
        edges = mx.nd.shuffle(seed)
        for sample_idx in range(0, (num_edges + args.minibatch_size - 1) // args.minibatch_size):
            edge_ids = edges[sample_idx * args.minibatch_size: (sample_idx + 1) * args.minibatch_size if (sample_idx + 1) * args.minibatch_size < num_edges else num_edges]
            #edge_ids = edges[0: args.minibatch_size]
            head_ids, tail_ids = dataset.train_dec_graph.find_edges(edge_ids)

            head_subgraphs = {}
            tail_subgraphs = {}
            for i, _ in enumerate(args.rating_vals):
                t = enc_graph.canonical_etypes[i * 2]
                rev_t = enc_graph.canonical_etypes[i * 2 + 1]

                head_in_edges = enc_graph.in_edges(head_ids, 'eid', etype=rev_t)
                tail_in_edges = enc_graph.in_edges(tail_ids, 'eid', etype=t)

                if head_in_edges.shape[0] == 0:
                    print('skip {} with 0'.format(rev_t))
                else:
                    head_subgraphs[rev_t] = head_in_edges

                if tail_in_edges.shape[0] == 0:
                    print('skip {} with 0'.format(t))
                else:
                    tail_subgraphs[t] = tail_in_edges

            head_subgraph = enc_graph.edge_subgraph(head_subgraphs)
            tail_subgraph = enc_graph.edge_subgraph(tail_subgraphs)
            edge_ids = edge_ids.as_in_context(args.ctx)
            true_relation_ratings = train_truths[edge_ids]
            true_relation_labels = train_labels[edge_ids]

            head_NID = head_subgraph.nodes['user'].data[dgl.NID]
            tail_NID = tail_subgraph.nodes['movie'].data[dgl.NID]

            g_user_fea[head_NID] = mx.nd.arange(head_NID.shape[0], dtype='int32')
            g_movie_fea[tail_NID] = mx.nd.arange(tail_NID.shape[0], dtype='int32')

            true_head_idx = g_user_fea[head_ids].as_in_context(args.ctx)
            true_tail_idx = g_movie_fea[tail_ids].as_in_context(args.ctx)

            with mx.autograd.record():
                pred_ratings = net(head_subgraph, tail_subgraph,
                                   true_head_idx, true_tail_idx)
                loss = rating_loss_net(pred_ratings, true_relation_labels).mean()
                loss.backward()
            gnorm = params_clip_global_norm(net.collect_params(), args.train_grad_clip, args.ctx)
            trainer.step(1.0, ignore_stale_grad=True)
            real_pred_ratings = (mx.nd.softmax(pred_ratings, axis=1) *
                             nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
            rmse = mx.nd.square(real_pred_ratings - true_relation_ratings).mean().asscalar()
            rmse = np.sqrt(rmse)
            loss = loss.asscalar()
            if sample_idx % 100 == 0:
                train_loss_logger.log(iter=iter_idx, idx=sample_idx,
                                  loss=loss, rmse=rmse)
                print("Iter={}, sample_idx={}, gnorm={:.3f}, loss={:.4f}, rmse={:.4f}".format(iter_idx,
                    sample_idx, gnorm, loss, rmse))

            gc.collect()

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (gluon_total_param_num(net)))
            print(gluon_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        if iter_idx % args.train_log_interval == 0:
           logging_str = "Iter={}, time={:.4f}".format(
                iter_idx, np.average(dur))

        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args, net=net, dataset=dataset, segment='valid')
            valid_loss_logger.log(iter = iter_idx, rmse = valid_rmse)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                #net.save_parameters(filename=os.path.join(args.save_dir, 'best_valid_net{}.params'.format(args.save_id)))
                test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test')
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                    and trainer.learning_rate <= args.train_min_lr:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train_decay_patience:
                    new_lr = max(trainer.learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                    if new_lr < trainer.learning_rate:
                        logging.info("\tChange the LR to %g" % new_lr)
                        trainer.set_learning_rate(new_lr)
                        no_better_valid = 0
        if iter_idx  % args.train_log_interval == 0:
            print(logging_str)
    print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='Run the baseline method.')

    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--ctx', dest='ctx', default='gpu0', type=str,
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')

    parser.add_argument('--data_name', default='ml-1m', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)

    #parser.add_argument('--model_remove_rating', type=bool, default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)

    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)

    # parser.add_argument('--train_rating_batch_size', type=int, default=10000)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--share_param', default=False, action='store_true')

    parser.add_argument('--minibatch_size', type=int, default=1000)

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]


    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + str(int(time.time()))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    args = config()
    np.random.seed(args.seed)
    mx.random.seed(args.seed, args.ctx)
    train(args)
