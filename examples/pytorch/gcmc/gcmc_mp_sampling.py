"""Training script"""
import os, time
import argparse
import logging
import random
import string
import traceback
import numpy as np
import torch as th
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from _thread import start_new_thread
from functools import wraps
from data import MovieLens
from model import SampleGCMCLayer, GCMCLayer, SampleBiDecoder, BiDecoder
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger
import dgl

class GCMCSampler:
    """ GCMCSampler
    """
    def __init__(self, num_edges, minibatch_size, dataset):
        self.num_edges = num_edges
        self.minibatch_size = minibatch_size
        self.seed = th.randperm(num_edges)
        self.dataset = dataset
        self.sample_idx = 0
        self.train_labels = dataset.train_labels
        self.train_truths = dataset.train_truths

        self.train_user_ids = th.zeros((dataset.train_enc_graph.number_of_nodes('user'),), dtype=th.int64)
        self.train_movie_ids = th.zeros((dataset.train_enc_graph.number_of_nodes('movie'),), dtype=th.int64)

    def sample_blocks(self, seeds):
        dataset = self.dataset
        edge_ids = seeds
        # generate frontiners for user and item
        possible_rating_values = dataset.possible_rating_values
        true_relation_ratings = self.train_truths[edge_ids]
        true_relation_labels = self.train_labels[edge_ids]

        head_id, tail_id = dataset.train_dec_graph.find_edges(edge_ids)
        utype, _, vtype = dataset.train_enc_graph.canonical_etypes[0]
        subg = []
        true_rel_ratings = []
        true_rel_labels = []
        for possible_rating_value in possible_rating_values:
            idx_loc = (true_relation_ratings == possible_rating_value)
            head = head_id[idx_loc]
            tail = tail_id[idx_loc]
            true_rel_ratings.append(true_relation_ratings[idx_loc])
            true_rel_labels.append(true_relation_labels[idx_loc])
            subg.append(dgl.bipartite((head, tail),
                                        utype=utype,
                                        etype=str(possible_rating_value),
                                        vtype=vtype,
                                        card=(dataset.train_enc_graph.number_of_nodes(utype),
                                              dataset.train_enc_graph.number_of_nodes(vtype))))
        
        g = dgl.hetero_from_relations(subg)
        g = dgl.compact_graphs(g)

        seed_nodes = {}
        for ntype in g.ntypes:
            seed_nodes[ntype] = g.nodes[ntype].data[dgl.NID]

        frontiner = dgl.in_subgraph(dataset.train_enc_graph, seed_nodes)
        frontiner = dgl.to_block(frontiner, seed_nodes)

        # copy from parent
        frontiner.dstnodes['user'].data['ci'] = \
            dataset.train_enc_graph.nodes['user'].data['ci'][frontiner.dstnodes['user'].data[dgl.NID]]
        frontiner.srcnodes['movie'].data['cj'] = \
            dataset.train_enc_graph.nodes['movie'].data['cj'][frontiner.srcnodes['movie'].data[dgl.NID]]
        frontiner.srcnodes['user'].data['cj'] = \
            dataset.train_enc_graph.nodes['user'].data['cj'][frontiner.srcnodes['user'].data[dgl.NID]]
        frontiner.dstnodes['movie'].data['ci'] = \
            dataset.train_enc_graph.nodes['movie'].data['ci'][frontiner.dstnodes['movie'].data[dgl.NID]]

        # handle features
        head_feat = frontiner.srcnodes['user'].data[dgl.NID].long() \
                    if dataset.user_feature is None else \
                       dataset.user_feature[frontiner.srcnodes['user'].data[dgl.NID]]
        tail_feat = frontiner.srcnodes['movie'].data[dgl.NID].long()\
                    if dataset.movie_feature is None else \
                       dataset.movie_feature[frontiner.srcnodes['movie'].data[dgl.NID]]

        true_rel_labels = th.cat(true_rel_labels, dim=0)
        true_rel_ratings = th.cat(true_rel_ratings, dim=0)
        return (g, frontiner, head_feat, tail_feat, true_rel_labels, true_rel_ratings)

class Net(nn.Module):
    def __init__(self, args, dev_id):
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
                                       share_user_item_param=args.share_param,
                                       device=dev_id)
        if args.mix_cpu_gpu and args.use_one_hot_fea:
            # if use_one_hot_fea, user and movie feature is None
            # W can be extremely large, with mix_cpu_gpu W should be stored in CPU
            self.encoder.partial_to(dev_id)
        else:
            self.encoder.to(dev_id)

        self.decoder = SampleBiDecoder(args.rating_vals,
                                       in_units=args.gcn_out_units,
                                       num_basis_functions=args.gen_r_num_basis_func)
        self.decoder.to(dev_id)

    def forward(self, compact_g, frontier, ufeat, ifeat, possible_rating_values):
        user_out, movie_out = self.encoder(frontier, frontier, ufeat, ifeat)

        head_emb = []
        tail_emb = []
        for possible_rating_value in possible_rating_values:
            head, tail = compact_g.all_edges(etype=str(possible_rating_value))
            head_emb.append(user_out[head])
            tail_emb.append(movie_out[tail])

        head_emb = th.cat(head_emb, dim=0)
        tail_emb = th.cat(tail_emb, dim=0)

        pred_ratings = self.decoder(head_emb, tail_emb)
        return pred_ratings

def evaluate(args, dev_id, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(dev_id)

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
    true_rel_ratings = []
    for sample_idx in range(0, (num_edges + args.minibatch_size - 1) // args.minibatch_size):
        net.eval()
        start = sample_idx * args.minibatch_size
        end = (sample_idx + 1) * args.minibatch_size \
              if (sample_idx + 1) * args.minibatch_size < num_edges else num_edges
        edge_ids = seed[start:end]

        sub_rating_values = rating_values[edge_ids]
        head_id, tail_id = dec_graph.find_edges(edge_ids)
        utype, _, vtype = enc_graph.canonical_etypes[0]
        
        subg = []
        sub_true_rel_ratings = []
        for possible_rating_value in possible_rating_values:
            idx_loc = (sub_rating_values == possible_rating_value)
            head = head_id[idx_loc]
            tail = tail_id[idx_loc]
            sub_true_rel_ratings.append(sub_rating_values[idx_loc])
            subg.append(dgl.bipartite((head, tail),
                                        utype=utype,
                                        etype=str(possible_rating_value),
                                        vtype=vtype,
                                        card=(enc_graph.number_of_nodes(utype),
                                              enc_graph.number_of_nodes(vtype))))

        g = dgl.hetero_from_relations(subg)
        g = dgl.compact_graphs(g)

        seed_nodes = {}
        for ntype in g.ntypes:
            seed_nodes[ntype] = g.nodes[ntype].data[dgl.NID]
        frontiner = dgl.in_subgraph(enc_graph, seed_nodes)
        frontiner = dgl.to_block(frontiner, seed_nodes)

        # copy from parent
        frontiner.dstnodes['user'].data['ci'] = \
            enc_graph.nodes['user'].data['ci'][frontiner.dstnodes['user'].data[dgl.NID]]
        frontiner.srcnodes['movie'].data['cj'] = \
            enc_graph.nodes['movie'].data['cj'][frontiner.srcnodes['movie'].data[dgl.NID]]
        frontiner.srcnodes['user'].data['cj'] = \
            enc_graph.nodes['user'].data['cj'][frontiner.srcnodes['user'].data[dgl.NID]]
        frontiner.dstnodes['movie'].data['ci'] = \
            enc_graph.nodes['movie'].data['ci'][frontiner.dstnodes['movie'].data[dgl.NID]]

        # handle features
        head_feat = frontiner.srcnodes['user'].data[dgl.NID].long() \
                    if dataset.user_feature is None else \
                       dataset.user_feature[frontiner.srcnodes['user'].data[dgl.NID]]
        tail_feat = frontiner.srcnodes['movie'].data[dgl.NID].long()\
                    if dataset.movie_feature is None else \
                       dataset.movie_feature[frontiner.srcnodes['movie'].data[dgl.NID]]
        head_feat = head_feat.to(dev_id)
        tail_feat = tail_feat.to(dev_id)
        sub_true_rel_ratings = th.cat(sub_true_rel_ratings, dim=0)
        with th.no_grad():
            pred_ratings = net(g, frontiner,
                               head_feat, tail_feat, possible_rating_values)
        batch_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                         nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings.append(batch_pred_ratings)
        true_rel_ratings.append(sub_true_rel_ratings)

    real_pred_ratings = th.cat(real_pred_ratings, dim=0)
    true_rel_ratings = th.cat(true_rel_ratings, dim=0).to(dev_id)
    rmse = ((real_pred_ratings - true_rel_ratings) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    return rmse

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    for etype in g.canonical_etypes:
        g.in_degree(0, etype=etype)
        g.out_degree(0, etype=etype)
        g.find_edges([0], etype=etype)

def config():
    parser = argparse.ArgumentParser(description='GCMC')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--gpu', type=str, default='0')
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
    parser.add_argument('--train_max_epoch', type=int, default=70)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=5)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--mix_cpu_gpu', default=False, action='store_true')
    parser.add_argument('--minibatch_size', type=int, default=10000)
    parser.add_argument('--num_workers_per_gpu', type=int, default=8)

    args = parser.parse_args()
    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args

@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset):
    dev_id = devices[proc_id]
    train_labels = dataset.train_labels
    train_truths = dataset.train_truths
    num_edges = train_truths.shape[0]
    sampler = GCMCSampler(num_edges,
                          args.minibatch_size,
                          dataset)

    seeds = th.arange(num_edges)
    dataloader = DataLoader(
        dataset=seeds,
        batch_size=args.minibatch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers_per_gpu)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)
    th.cuda.set_device(dev_id)

    nd_possible_rating_values = \
        th.FloatTensor(dataset.possible_rating_values)
    nd_possible_rating_values = nd_possible_rating_values.to(dev_id)

    net = Net(args=args, dev_id=dev_id)
    net = net.to(dev_id)
    if n_gpus > 1:
        net = DistributedDataParallel(net, device_ids=[dev_id], output_device=dev_id)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_epoch = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0
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
        if epoch > 1:
            t0 = time.time()
        net.train()
        for step, sample_data in enumerate(dataloader):           
            compact_g, frontier, head_feat, tail_feat, \
                true_relation_labels, true_relation_ratings = sample_data
            head_feat = head_feat.to(dev_id)
            tail_feat = tail_feat.to(dev_id)

            pred_ratings = net(compact_g, frontier, head_feat, tail_feat, dataset.possible_rating_values)
            loss = rating_loss_net(pred_ratings, true_relation_labels.to(dev_id)).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
            optimizer.step()

            if proc_id == 0 and iter_idx == 1:
                print("Total #Param of net: %d" % (torch_total_param_num(net)))

            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - true_relation_ratings.to(dev_id)) ** 2).sum()
            count_rmse += rmse.item()
            count_num += pred_ratings.shape[0]

            if iter_idx % args.train_log_interval == 0:
                logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}".format(
                    iter_idx, count_loss/iter_idx, count_rmse/count_num)
                count_rmse = 0
                count_num = 0

            if iter_idx % args.train_log_interval == 0:
                print("[{}] {}".format(proc_id, logging_str))

            iter_idx += 1
        if epoch > 1:
            epoch_time = time.time() - t0
            print("Epoch {} time {}".format(epoch, epoch_time))

        if epoch % args.train_valid_interval == 0:
            if n_gpus > 1:
                th.distributed.barrier()
            if proc_id == 0:
                valid_rmse = evaluate(args=args, dev_id=dev_id, net=net, dataset=dataset, segment='valid')
                logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_epoch = epoch
                    test_rmse = evaluate(args=args, dev_id=dev_id, net=net, dataset=dataset, segment='test')
                    best_test_rmse = test_rmse
                    logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
                else:
                    no_better_valid += 1
                    if no_better_valid > args.train_decay_patience:
                        new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                        if new_lr < learning_rate:
                            logging.info("\tChange the LR to %g" % new_lr)
                            learning_rate = new_lr
                            for p in optimizer.param_groups:
                                p['lr'] = learning_rate
                            no_better_valid = 0
                            print("Change the LR to %g" % new_lr)
            # sync on evalution
            if n_gpus > 1:
                th.distributed.barrier()

        print(logging_str)
    if proc_id == 0:
        print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
              best_epoch, best_valid_rmse, best_test_rmse))

if __name__ == '__main__':
    args = config()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # For GCMC based on sampling, we require node has its own features.
    # Otherwise (node_id is the feature), the model can not scale
    dataset = MovieLens(args.data_name,
                        'cpu',
                        mix_cpu_gpu=args.mix_cpu_gpu,
                        use_one_hot_fea=args.use_one_hot_fea,
                        symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio,
                        valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    if n_gpus == 1:
        run(0, n_gpus, args, devices, dataset)
    else:
        prepare_mp(dataset.train_enc_graph)
        prepare_mp(dataset.train_dec_graph)
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, dataset))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
