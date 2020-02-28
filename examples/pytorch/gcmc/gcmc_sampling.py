"""Training script"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import torch as th
import torch.nn as nn
from data import MovieLens
from model import SampleGCMCLayer, GCMCLayer, SampleBiDecoder, BiDecoder
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger
import dgl

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.encoder = SampleGCMCLayer(args.rating_vals,
                                       args.src_in_units,
                                       args.dst_in_units,
                                       args.gcn_agg_units,
                                       args.gcn_out_units,
                                       args.gcn_dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param)
        if args.mix_cpu_gpu and args.use_one_hot_fea:
            # if use_one_hot_fea, user and movie feature is None
            # W can be extremely large, with mix_cpu_gpu W should be stored in CPU
            self.encoder.partial_to(args.device)
        else:
            self.encoder.to(args.device)

        self.decoder = SampleBiDecoder(args.rating_vals,
                                       in_units=args.gcn_out_units,
                                       num_basis_functions=args.gen_r_num_basis_func)
        self.decoder.to(args.device)

    def forward(self, head_enc, tail_enc, ufeat, ifeat, head_id, tail_id):
        user_out, movie_out = self.encoder(
            head_enc,
            tail_enc,
            ufeat,
            ifeat)

        user_out = user_out[head_id]
        movie_out = movie_out[tail_id]
        
        pred_ratings = self.decoder(user_out, movie_out)
        return pred_ratings

def evaluate(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)

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
    seed = th.arange(num_edges)
    count_rmse = 0
    count_num = 0
    real_pred_ratings = []
    for sample_idx in range(0, (num_edges + args.minibatch_size - 1) // args.minibatch_size):
        net.eval()
        edge_ids = seed[sample_idx * args.minibatch_size: \
                        (sample_idx + 1) * args.minibatch_size \
                        if (sample_idx + 1) * args.minibatch_size < num_edges \
                        else num_edges]

        head_id, tail_id = dec_graph.find_edges(edge_ids)
        head_type, _, tail_type = dec_graph.canonical_etypes[0]
        heads = {head_type : th.unique(head_id)}
        tails = {tail_type : th.unique(tail_id)}

        head_frontier = dgl.in_subgraph(enc_graph, heads)
        tail_frontier = dgl.in_subgraph(enc_graph, tails)

        head_frontier = dgl.compact_graphs(head_frontier)
        tail_frontier = dgl.compact_graphs(tail_frontier)
        head_frontier.nodes['user'].data['ci'] = \
            enc_graph.nodes['user'].data['ci'][head_frontier.nodes['user'].data[dgl.NID]]
        head_frontier.nodes['movie'].data['cj'] = \
            enc_graph.nodes['movie'].data['cj'][head_frontier.nodes['movie'].data[dgl.NID]]
        tail_frontier.nodes['user'].data['cj'] = \
            enc_graph.nodes['user'].data['cj'][tail_frontier.nodes['user'].data[dgl.NID]]
        tail_frontier.nodes['movie'].data['ci'] = \
            enc_graph.nodes['movie'].data['ci'][tail_frontier.nodes['movie'].data[dgl.NID]]

        enc_graph.nodes['user'].data['ids'][head_frontier.nodes['user'].data[dgl.NID]] = \
            th.arange(head_frontier.nodes['user'].data[dgl.NID].shape[0])
        enc_graph.nodes['movie'].data['ids'][tail_frontier.nodes['movie'].data[dgl.NID]] = \
            th.arange(tail_frontier.nodes['movie'].data[dgl.NID].shape[0])

        head_feat = tail_frontier.nodes['user'].data[dgl.NID].long().to(args.device) \
                    if dataset.user_feature is None else \
                       dataset.user_feature[tail_frontier.nodes['user'].data[dgl.NID]]
        tail_feat = head_frontier.nodes['movie'].data[dgl.NID].long().to(args.device) \
                    if dataset.movie_feature is None else \
                        dataset.movie_feature[head_frontier.nodes['movie'].data[dgl.NID]]

        head_id = enc_graph.nodes['user'].data['ids'][head_id]
        tail_id = enc_graph.nodes['movie'].data['ids'][tail_id]
        with th.no_grad():
            pred_ratings = net(head_frontier, tail_frontier,
                               head_feat, tail_feat,
                               head_id, tail_id)
        batch_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                         nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings.append(batch_pred_ratings)
    real_pred_ratings = th.cat(real_pred_ratings, dim=0)
    rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    return rmse

def train(args):
    print(args)
    dataset = MovieLens(args.data_name,
                        args.device,
                        mix_cpu_gpu=args.mix_cpu_gpu,
                        use_one_hot_fea=args.use_one_hot_fea,
                        symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio,
                        valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    ### build the net
    net = Net(args=args)
    #net = net.to(args.device)
    nd_possible_rating_values = \
        th.FloatTensor(dataset.possible_rating_values).to(args.device)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

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
    count_rmse = 0
    count_num = 0
    count_loss = 0
    train_labels = dataset.train_labels
    train_truths = dataset.train_truths
    dataset.train_enc_graph.nodes['user'].data['ids'] = \
        th.zeros((dataset.train_enc_graph.number_of_nodes('user'),), dtype=th.int64)
    dataset.train_enc_graph.nodes['movie'].data['ids'] = \
        th.zeros((dataset.train_enc_graph.number_of_nodes('movie'),), dtype=th.int64)
    dataset.valid_enc_graph.nodes['user'].data['ids'] = \
        th.zeros((dataset.valid_enc_graph.number_of_nodes('user'),), dtype=th.int64)
    dataset.valid_enc_graph.nodes['movie'].data['ids'] = \
        th.zeros((dataset.valid_enc_graph.number_of_nodes('movie'),), dtype=th.int64)
    dataset.test_enc_graph.nodes['user'].data['ids'] = \
        th.zeros((dataset.test_enc_graph.number_of_nodes('user'),), dtype=th.int64)
    dataset.test_enc_graph.nodes['movie'].data['ids'] = \
        th.zeros((dataset.test_enc_graph.number_of_nodes('movie'),), dtype=th.int64)
    print("Start training ...")
    dur = []
    iter_idx = 1
    for epoch in range(1, args.train_max_epoch):
        num_edges = train_truths.shape[0]
        seed = th.randperm(num_edges)

        if epoch > 1:
            t0 = time.time()
        for sample_idx in range(0, (num_edges + args.minibatch_size - 1) // args.minibatch_size):
            net.train()
            edge_ids = seed[sample_idx * args.minibatch_size: \
                             (sample_idx + 1) * args.minibatch_size \
                             if (sample_idx + 1) * args.minibatch_size < num_edges \
                             else num_edges]

            true_relation_ratings = train_truths[edge_ids]
            true_relation_labels = train_labels[edge_ids]
            head_id, tail_id = dataset.train_dec_graph.find_edges(edge_ids)
            head_type, _, tail_type = dataset.train_dec_graph.canonical_etypes[0]
            heads = {head_type : th.unique(head_id)}
            tails = {tail_type : th.unique(tail_id)}
            head_frontier = dgl.in_subgraph(dataset.train_enc_graph, heads)
            tail_frontier = dgl.in_subgraph(dataset.train_enc_graph, tails)

            head_frontier = dgl.compact_graphs(head_frontier)
            tail_frontier = dgl.compact_graphs(tail_frontier)
            head_frontier.nodes['user'].data['ci'] = \
                dataset.train_enc_graph.nodes['user'].data['ci'][head_frontier.nodes['user'].data[dgl.NID]]
            head_frontier.nodes['movie'].data['cj'] = \
                dataset.train_enc_graph.nodes['movie'].data['cj'][head_frontier.nodes['movie'].data[dgl.NID]]
            tail_frontier.nodes['user'].data['cj'] = \
                dataset.train_enc_graph.nodes['user'].data['cj'][tail_frontier.nodes['user'].data[dgl.NID]]
            tail_frontier.nodes['movie'].data['ci'] = \
                dataset.train_enc_graph.nodes['movie'].data['ci'][tail_frontier.nodes['movie'].data[dgl.NID]]

            dataset.train_enc_graph.nodes['user'].data['ids'][head_frontier.nodes['user'].data[dgl.NID]] = \
                th.arange(head_frontier.nodes['user'].data[dgl.NID].shape[0])
            dataset.train_enc_graph.nodes['movie'].data['ids'][tail_frontier.nodes['movie'].data[dgl.NID]] = \
                th.arange(tail_frontier.nodes['movie'].data[dgl.NID].shape[0])

            head_feat = tail_frontier.nodes['user'].data[dgl.NID].long().to(args.device) \
                        if dataset.user_feature is None else \
                           dataset.user_feature[tail_frontier.nodes['user'].data[dgl.NID]]
            tail_feat = head_frontier.nodes['movie'].data[dgl.NID].long().to(args.device) \
                        if dataset.movie_feature is None else \
                            dataset.movie_feature[head_frontier.nodes['movie'].data[dgl.NID]]
            head_feat = head_feat.to(args.device)
            tail_feat = tail_feat.to(args.device)
            head_id = dataset.train_enc_graph.nodes['user'].data['ids'][head_id]
            tail_id = dataset.train_enc_graph.nodes['movie'].data['ids'][tail_id]

            pred_ratings = net(head_frontier, tail_frontier, head_feat, tail_feat, head_id, tail_id)
            loss = rating_loss_net(pred_ratings, true_relation_labels).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
            optimizer.step()

            if iter_idx == 1:
                print("Total #Param of net: %d" % (torch_total_param_num(net)))
                print(torch_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - true_relation_ratings) ** 2).sum()
            count_rmse += rmse.item()
            count_num += pred_ratings.shape[0]

            if iter_idx % args.train_log_interval == 0:
                train_loss_logger.log(iter=iter_idx,
                                    loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
                logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}".format(
                    iter_idx, count_loss/iter_idx, count_rmse/count_num)
                count_rmse = 0
                count_num = 0

            if iter_idx  % args.train_log_interval == 0:
                print(logging_str)

            iter_idx += 1

        if epoch > 1:
            epoch_time = time.time() - t0
            print("Epoch {} time {}".format(epoch, epoch_time))

        if epoch % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args, net=net, dataset=dataset, segment='valid')
            valid_loss_logger.log(iter = iter_idx, rmse = valid_rmse)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test')
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                    and learning_rate <= args.train_min_lr:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train_decay_patience:
                    new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                    if new_lr < learning_rate:
                        logging.info("\tChange the LR to %g" % new_lr)
                        learning_rate = new_lr
                        for p in optimizer.param_groups:
                            p['lr'] = learning_rate
                        no_better_valid = 0
                        print("Change the LR to %g" % new_lr)
        print(logging_str)

    print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='GCMC')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--data_name', default='ml-1m', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)
    parser.add_argument('--train_max_epoch', type=int, default=200)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=20)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--mix_cpu_gpu', default=False, action='store_true')
    parser.add_argument('--minibatch_size', type=int, default=10000)

    args = parser.parse_args()
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')

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
    args = config()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)