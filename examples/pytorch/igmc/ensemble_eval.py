"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import torch as th
import torch.nn as nn
from data import MovieLens
from model import IGMC
from dataset import MovieLensDataset, collate_movielens 
from torch.utils.data import DataLoader
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger

def ensemble_evaluate(args, net, ckpts_path):
    # Evaluate RMSE
    # recreate dataset everytime
    dataset_base = MovieLens(args.data_name, args.device, args, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm,
                            test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)

    test_dataset = MovieLensDataset(dataset_base.test_graphs, args.device, mode='test', link_dropout=args.link_dropout, force_undirected=args.force_undirected)

    data_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_movielens)


    net.eval()
    preds = []
    ys = []
    for i, ckpt in enumerate(ckpts_path):
        net.load_state_dict(th.load(ckpt))
        print (ckpt)
        pred = []
        res = []
        for idx, batch in enumerate(data_iter):
            graph = batch[0]
            rating_gt = batch[1]
            if i == 0:
                ys.append(rating_gt)
            with th.no_grad():
                pred.append(net(graph))
            print (idx)
        pred = th.cat(pred, 0).view(-1, 1)
        preds.append(pred)
    ys = th.cat(ys, 0)
    preds = th.cat(preds, 1).mean(1)
    print (ys.shape, preds.shape)
    rmse = ((preds - ys) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    print ('ensemble rmse', rmse)
    return rmse

def eval(args):
    ### build the net
    # multiply num_relations by 2 because now we have two types for u-v edge and v-u edge
    # NOTE: can't remember why we need add 2 for dim, need check
    net = IGMC(in_dim=args.hop*2+1+1, num_relations=5, num_bases=args.num_igmc_bases, regression=True)
    net = net.to(args.device)

    #rating_loss_net = nn.CrossEntropyLoss()
    rating_loss_net = nn.MSELoss()

    # save model for each epoch
    #start_epoch, end_epoch, interval = args.train_max_epoch-30-1, args.train_max_epoch, 10

    ckpt_idxs = list(map(int, args.ckpt_idxs.split(',')))
    ckpts_path = [os.path.join(args.save_dir, 'ckpt_%d.pt' % epoch_idx) for epoch_idx in ckpt_idxs]
    print (ckpts_path)


    test_rmse = ensemble_evaluate(args=args, net=net, ckpts_path=ckpts_path)


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
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=100)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--share_param', default=False, action='store_true')
    # igmc settings
    parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number')
    parser.add_argument('--arr_lambda', type=float, default=0.001)
    parser.add_argument('--num_igmc_bases', type=int, default=2)
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max_nodes_per_hop', type=int, default=10000, 
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    parser.add_argument('--use_features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    parser.add_argument('--train_max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--train_decay_epoch', type=int, default=50)
    parser.add_argument('--link-dropout', type=float, default=0.2, help='link dropout rate')
    parser.add_argument('--force-undirected', action='store_true', default=False, help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    parser.add_argument('--train_val', action='store_true', default=False)
    # ensemble different ckpt
    parser.add_argument('--ckpt_idxs', default='-1', type=str, help='ckpts to use for ensemble evaluation')
    # edge dropout settings
    parser.add_argument('--adj_dropout', type=float, default=0.2, 
                    help='if not 0, random drops edges from adjacency matrix with this prob')

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
    eval(args)
