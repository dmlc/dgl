"""Training IGMC model on the MovieLens dataset."""

import os
import time
import argparse
import random
import string
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import IGMC
from data import MovieLens
from dataset import MovieLensDataset, collate_movielens 
from utils import torch_total_param_num, torch_net_info, MetricLogger, links2subgraphs

def evaluate(args, model, data_iter):
    # Evaluate RMSE
    model.eval()
    res = []
    for idx, batch in enumerate(data_iter):
        graph = batch[0]
        rating_gt = batch[1]
        with th.no_grad():
            pred_ratings = model(graph)
        rmse = ((pred_ratings - rating_gt) ** 2.).mean().item()
        res.append(rmse)
    return np.sqrt(np.mean(res))


def adj_rating_reg(model):
    arr_loss = 0
    for conv in model.convs:
        weight = conv.weight.view(conv.num_bases, conv.in_feat * conv.out_feat)
        weight = th.matmul(conv.w_comp, weight).view(conv.num_rels, conv.in_feat, conv.out_feat)
        arr_loss += th.sum((weight[1:, :, :] - weight[:-1, :, :])**2)
    return arr_loss


def train(args):
    # import multiprocessing as mp
    # pool = mp.Pool(mp.cpu_count())
    
    ### build the model
    dataset_base = MovieLens(args.data_name, use_features=args.use_features, train_val=args.train_val,
                             test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)

    # multiply num_relations by 2 because now we have two types for u-v edge and v-u edge
    # NOTE: can't remember why we need add 2 for dim, need check
    model = IGMC(in_feats=(args.hop+1)*2,
                 latent_dim=[32, 32, 32, 32],
                 num_relations=dataset_base.num_edge_types, 
                 num_bases=4, 
                 regression=True, 
                 edge_drop=args.edge_dropout,
                #  side_features=args.use_features,
                #  n_side_features=n_features,
                #  multiply_by=args.multiply_by
            ).to(args.device)
    rating_loss_fn = nn.MSELoss()
    learning_rate = args.train_lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### prepare the logger
    train_loss_logger = MetricLogger(['epoch', 'iter', 'loss', 'rmse'], ['%d', '%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['epoch', 'rmse'], ['%d', '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['epoch', 'rmse'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    ### declare the loss information
    best_valid_rmse = np.inf
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0

    print("Start training ...")
    dur = []
    for epoch_idx in range(args.train_max_epoch):
        print ('Epoch', epoch_idx)

        # Rebuild subgraph for each epoch to force 
        train_graphs, val_graphs, test_graphs = links2subgraphs(
                dataset_base.rating_mx_train, dataset_base.rating_values, # pool,
                dataset_base.train_rating_pairs, dataset_base.valid_rating_pairs, dataset_base.test_rating_pairs,
                dataset_base.train_rating_values, dataset_base.valid_rating_values, dataset_base.test_rating_values,
                args.hop, args.sample_ratio, args.max_nodes_per_hop, max_node_label=args.hop*2+1,
                user_feature=dataset_base.user_feature, movie_feature=dataset_base.movie_feature,
                train_val=args.train_val, parallel=True)
        train_dataset = MovieLensDataset(train_graphs, args.device, mode='train', link_dropout=args.link_dropout, force_undirected=args.force_undirected)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_movielens)
        val_dataset = MovieLensDataset(val_graphs, args.device, mode='test', link_dropout=args.link_dropout, force_undirected=args.force_undirected)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_movielens)
        test_dataset = MovieLensDataset(test_graphs, args.device, mode='test', link_dropout=args.link_dropout, force_undirected=args.force_undirected)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_movielens)

        for iter_idx, batch in enumerate(train_loader):
            if iter_idx > 3:
                t0 = time.time()

            model.train()
            pred_ratings = model(batch[0])
            train_gt_labels = batch[1]
            loss = rating_loss_fn(pred_ratings, train_gt_labels).mean() + args.arr_lambda * adj_rating_reg(model)
            count_loss += loss.item() * pred_ratings.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_idx > 3:
                dur.append(time.time() - t0)

            if iter_idx == 1:
                print("Total #Param of model: %d" % (torch_total_param_num(model)))
                print(torch_net_info(model, save_path=os.path.join(args.save_dir, 'model%d.txt' % args.save_id)))

            rmse = ((pred_ratings - train_gt_labels) ** 2).sum()
            count_rmse += rmse.item()
            count_num += pred_ratings.shape[0]

            if iter_idx % args.train_log_interval == 0:
                train_loss_logger.log(epoch=epoch_idx, iter=iter_idx,
                                      loss=count_loss/(count_num), rmse=np.sqrt(count_rmse/count_num))
                logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}, time={:.4f}".format(
                    iter_idx, count_loss/count_num, count_rmse/count_num,
                    np.average(dur))
                print (logging_str)
                count_rmse = 0
                count_num = 0
                count_loss = 0

        # save model for each epoch
        ckpt_path = os.path.join(args.save_dir, 'ckpt_%d.pt' % epoch_idx)
        th.save(model.state_dict(), ckpt_path)

        valid_rmse = evaluate(args=args, model=model, data_iter=val_loader)
        valid_loss_logger.log(epoch = epoch_idx, rmse = valid_rmse)
        logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

        if best_valid_rmse > valid_rmse:
            best_valid_rmse = valid_rmse
            best_iter = iter_idx
        test_rmse = evaluate(args=args, model=model, data_iter=test_loader)
        best_test_rmse = test_rmse
        test_loss_logger.log(epoch=epoch_idx, rmse=test_rmse)
        logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
        
        if epoch_idx + 1 % args.train_decay_epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] = args.train_lr_decay_factor * p['lr']

        print (logging_str)
    print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))
    # pool.close()
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()

def config():
    parser = argparse.ArgumentParser(description='IGMC')

    # general settings
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')

    parser.add_argument('--data_name', default='ml-100k', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) # for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.2)
    
    parser.add_argument('--train_log_interval', type=int, default=100)
    parser.add_argument('--train_valid_interval', type=int, default=1)

    
    # subgraph extraction settings
    parser.add_argument('--hop', default=1, metavar='S', 
                        help='enclosing subgraph hop number')
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max_nodes_per_hop', type=int, default=10000, 
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    parser.add_argument('--use_features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    # parser.add_argument('--use_one_hot_fea', action='store_true', default=True)
    
    # edge dropout settings
    parser.add_argument('--edge_dropout', type=float, default=0.2, 
                    help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force_undirected', action='store_true', default=False, 
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    
    # optimization settings
    # parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    # parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--arr_lambda', type=float, default=0.001)
    parser.add_argument('--num_igmc_bases', type=int, default=2)
    parser.add_argument('--train_max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--train_decay_epoch', type=int, default=50)
    parser.add_argument('--link-dropout', type=float, default=0.2, help='link dropout rate')
    parser.add_argument('--train_val', action='store_true', default=True)

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)
