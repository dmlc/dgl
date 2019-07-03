import os
import argparse
import logging
import random
import string
import numpy as np
import mxnet as mx
from mxnet import gluon
from data import MovieLens
from model import GCMCLayer, BiDecoder, InnerProductLayer
from utils import get_activation, parse_ctx, gluon_net_info, gluon_total_param_num, params_clip_global_norm, \
    logging_config, MetricLogger
from mxnet.gluon import nn, HybridBlock, Block

class Net(Block):
    def __init__(self, args, **kwargs):
        super(Net, self).__init__(**kwargs)
        self._act = get_activation(args.model_activation)
        with self.name_scope():
            self.encoder = GCMCLayer(src_key=args.src_key,
                                     dst_key=args.dst_key,
                                     src_in_units=args.src_in_units,
                                     dst_in_units=args.dst_in_units,
                                     agg_units=args.gcn_agg_units,
                                     out_units=args.gcn_out_units,
                                     num_links=args.nratings,
                                     dropout_rate=args.gcn_dropout,
                                     agg_accum=args.gcn_agg_accum,
                                     agg_act=args.model_activation,
                                     prefix='enc_')
            if args.gen_r_use_classification:
                self.gen_ratings = BiDecoder(in_units=args.gcn_out_units,
                                             out_units=args.nratings,
                                             num_basis_functions=args.gen_r_num_basis_func,
                                             prefix='gen_rating')
            else:
                self.gen_ratings = InnerProductLayer(prefix='gen_rating')


    def forward(self, graph, rating_node_pairs):
        # start = time.time()
        user_out, movie_out = self.encoder(graph)
        #print("The time for encoder is: {:.1f}s".format(time.time()-start))
        # Generate the predicted ratings
        #start = time.time()
        rating_user_fea = mx.nd.take(user_out, rating_node_pairs[0])
        rating_item_fea = mx.nd.take(movie_out, rating_node_pairs[1])
        pred_ratings = self.gen_ratings(rating_user_fea, rating_item_fea)
        #print("The time for decoder is: {:.1f}s".format(time.time()-start))
        return pred_ratings

def evaluate(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)

    if segment == "valid":
        rating_pairs = dataset.valid_rating_pairs
        rating_values = dataset.valid_rating_values
        graph = dataset.train_graph
    elif segment == "test":
        rating_pairs = dataset.test_rating_pairs
        rating_values = dataset.test_rating_values
        graph = dataset.test_graph
        #graph = dataset.train_graph
    else:
        raise NotImplementedError

    rating_pairs = mx.nd.array(rating_pairs, ctx=args.ctx, dtype=np.int64)
    rating_values = mx.nd.array(rating_values, ctx=args.ctx, dtype=np.float32)

    # Evaluate RMSE
    pred_ratings = net(graph, rating_pairs)
    if args.gen_r_use_classification:
        real_pred_ratings = (mx.nd.softmax(pred_ratings, axis=1) *
                             nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
        rmse = mx.nd.square(real_pred_ratings - rating_values).mean().asscalar()
    else:
        rating_mean = dataset.train_rating_values.mean()
        rating_std = dataset.train_rating_values.std()
        rmse = mx.nd.square(mx.nd.clip(pred_ratings.reshape((-1,)) * rating_std + rating_mean,
                                       possible_rating_values.min(),
                                       possible_rating_values.max()) - rating_values).mean().asscalar()
    rmse  = np.sqrt(rmse)
    return rmse

def train(args):
    dataset = MovieLens(args.data_name, args.ctx, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm)
    print("Loading data finished ...\n")

    args.src_key = dataset.name_user
    args.dst_key = dataset.name_movie
    args.src_in_units = dataset.user_feature.shape[1]
    args.dst_in_units = dataset.movie_feature.shape[1]
    args.nratings = dataset.possible_rating_values.size

    ### build the net
    net = Net(args=args)
    net.initialize(init=mx.init.Xavier(factor_type='in'), ctx=args.ctx)
    net.hybridize()
    if args.gen_r_use_classification:
        nd_possible_rating_values = mx.nd.array(dataset.possible_rating_values, ctx=args.ctx, dtype=np.float32)
        rating_loss_net = gluon.loss.SoftmaxCELoss()
    else:
        rating_mean = dataset.train_rating_values.mean()
        rating_std = dataset.train_rating_values.std()
        rating_loss_net = gluon.loss.L2Loss()
    rating_loss_net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), args.train_optimizer, {'learning_rate': args.train_lr})
    print("Loading network finished ...\n")

    ### perpare training data
    train_rating_pairs = mx.nd.array(dataset.train_rating_pairs, ctx=args.ctx, dtype=np.int64)
    train_gt_ratings = mx.nd.array(dataset.train_rating_values, ctx=args.ctx, dtype=np.float32)

    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss', 'rmse'], ['%d', '%.4f', '%.4f'],
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

    print("Start training ...")
    for iter_idx in range(1, args.train_max_iter):
        if args.gen_r_use_classification:
            train_gt_label = mx.nd.array(np.searchsorted(dataset.possible_rating_values, dataset.train_rating_values),
                                      ctx=args.ctx, dtype=np.int32)
        with mx.autograd.record():
            pred_ratings = net(dataset.train_graph, train_rating_pairs)
            if args.gen_r_use_classification:
                loss = rating_loss_net(pred_ratings, train_gt_label).mean()
            else:
                loss = rating_loss_net(mx.nd.reshape(pred_ratings, shape=(-1,)),
                                       (train_gt_ratings - rating_mean) / rating_std ).mean()
            #loss.wait_to_read()
            loss.backward()

        count_loss += loss.asscalar()
        gnorm = params_clip_global_norm(net.collect_params(), args.train_grad_clip, args.ctx)
        avg_gnorm += gnorm
        trainer.step(1.0) #, ignore_stale_grad=True)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (gluon_total_param_num(net)))
            print(gluon_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        if args.gen_r_use_classification:
            real_pred_ratings = (mx.nd.softmax(pred_ratings, axis=1) *
                                 nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
            rmse = mx.nd.square(real_pred_ratings - train_gt_ratings).sum()
        else:
            rmse = mx.nd.square(pred_ratings.reshape((-1,)) * rating_std + rating_mean - train_gt_ratings).sum()
        count_rmse += rmse.asscalar()
        count_num += pred_ratings.shape[0]

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
            logging_str = "Iter={}, gnorm={:.3f}, loss={:.4f}, rmse={:.4f}".format(
                iter_idx, avg_gnorm/args.train_log_interval, count_loss/iter_idx, count_rmse/count_num)
            avg_gnorm = 0
            count_rmse = 0
            count_num = 0

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
    parser.add_argument('--use_one_hot_fea', type=bool, default=True)

    #parser.add_argument('--model_remove_rating', type=bool, default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)

    parser.add_argument('--gen_r_use_classification', type=bool, default=True)
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

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]


    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    #os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = config()
    #logging_config(folder=args.save_dir, name='log', no_console=args.silent)
    ### TODO save the args
    np.random.seed(args.seed)
    mx.random.seed(args.seed, args.ctx)
    train(args)
