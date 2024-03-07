"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import argparse
import logging
import os
import random
import string
import time

import numpy as np
import torch as th
import torch.nn as nn
from data import MovieLens
from model import BiDecoder, GCMCLayer
from utils import (
    get_activation,
    get_optimizer,
    MetricLogger,
    torch_net_info,
    torch_total_param_num,
)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.encoder = GCMCLayer(
            args.rating_vals,
            args.src_in_units,
            args.dst_in_units,
            args.gcn_agg_units,
            args.gcn_out_units,
            args.gcn_dropout,
            args.gcn_agg_accum,
            agg_act=self._act,
            share_user_item_param=args.share_param,
            device=args.device,
        )
        self.decoder = BiDecoder(
            in_units=args.gcn_out_units,
            num_classes=len(args.rating_vals),
            num_basis=args.gen_r_num_basis_func,
        )

    def forward(self, enc_graph, dec_graph, ufeat, ifeat):
        user_out, movie_out = self.encoder(enc_graph, ufeat, ifeat)
        pred_ratings = self.decoder(dec_graph, user_out, movie_out)
        return pred_ratings


def evaluate(args, net, dataset, segment="valid"):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(
        args.device
    )

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

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings = net(
            enc_graph, dec_graph, dataset.user_feature, dataset.movie_feature
        )
    real_pred_ratings = (
        th.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)
    ).sum(dim=1)
    rmse = ((real_pred_ratings - rating_values) ** 2.0).mean().item()
    rmse = np.sqrt(rmse)
    return rmse


def train(args):
    print(args)
    dataset = MovieLens(
        args.data_name,
        args.device,
        use_one_hot_fea=args.use_one_hot_fea,
        symm=args.gcn_agg_norm_symm,
        test_ratio=args.data_test_ratio,
        valid_ratio=args.data_valid_ratio,
    )
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    ### build the net
    net = Net(args=args)
    net = net.to(args.device)
    nd_possible_rating_values = th.FloatTensor(
        dataset.possible_rating_values
    ).to(args.device)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(
        net.parameters(), lr=learning_rate
    )
    print("Loading network finished ...\n")

    ### perpare training data
    train_gt_labels = dataset.train_labels
    train_gt_ratings = dataset.train_truths

    ### prepare the logger
    train_loss_logger = MetricLogger(
        ["iter", "loss", "rmse"],
        ["%d", "%.4f", "%.4f"],
        os.path.join(args.save_dir, "train_loss%d.csv" % args.save_id),
    )
    valid_loss_logger = MetricLogger(
        ["iter", "rmse"],
        ["%d", "%.4f"],
        os.path.join(args.save_dir, "valid_loss%d.csv" % args.save_id),
    )
    test_loss_logger = MetricLogger(
        ["iter", "rmse"],
        ["%d", "%.4f"],
        os.path.join(args.save_dir, "test_loss%d.csv" % args.save_id),
    )

    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(args.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(args.device)

    print("Start training ...")
    dur = []
    for iter_idx in range(1, args.train_max_iter):
        if iter_idx > 3:
            t0 = time.time()
        net.train()
        pred_ratings = net(
            dataset.train_enc_graph,
            dataset.train_dec_graph,
            dataset.user_feature,
            dataset.movie_feature,
        )
        loss = rating_loss_net(pred_ratings, train_gt_labels).mean()
        count_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(
                torch_net_info(
                    net,
                    save_path=os.path.join(
                        args.save_dir, "net%d.txt" % args.save_id
                    ),
                )
            )

        real_pred_ratings = (
            th.softmax(pred_ratings, dim=1)
            * nd_possible_rating_values.view(1, -1)
        ).sum(dim=1)
        rmse = ((real_pred_ratings - train_gt_ratings) ** 2).sum()
        count_rmse += rmse.item()
        count_num += pred_ratings.shape[0]

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(
                iter=iter_idx,
                loss=count_loss / (iter_idx + 1),
                rmse=count_rmse / count_num,
            )
            logging_str = "Iter={:4d}, loss={:.4f}, rmse={:.4f}".format(
                iter_idx,
                count_loss / iter_idx,
                count_rmse / count_num,
            )
            if iter_idx > 3:
                logging_str += ", time={:.4f}".format(np.average(dur))

            count_rmse = 0
            count_num = 0

        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(
                args=args, net=net, dataset=dataset, segment="valid"
            )
            valid_loss_logger.log(iter=iter_idx, rmse=valid_rmse)
            logging_str += ",\tVal RMSE={:.4f}".format(valid_rmse)

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(
                    args=args, net=net, dataset=dataset, segment="test"
                )
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ", Test RMSE={:.4f}".format(test_rmse)
            else:
                no_better_valid += 1
                if (
                    no_better_valid > args.train_early_stopping_patience
                    and learning_rate <= args.train_min_lr
                ):
                    logging.info(
                        "Early stopping threshold reached. Stop training."
                    )
                    break
                if no_better_valid > args.train_decay_patience:
                    new_lr = max(
                        learning_rate * args.train_lr_decay_factor,
                        args.train_min_lr,
                    )
                    if new_lr < learning_rate:
                        learning_rate = new_lr
                        logging.info("\tChange the LR to %g" % new_lr)
                        for p in optimizer.param_groups:
                            p["lr"] = learning_rate
                        no_better_valid = 0
        if iter_idx % args.train_log_interval == 0:
            print(logging_str)
    print(
        "Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}".format(
            best_iter, best_valid_rmse, best_test_rmse
        )
    )
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description="GCMC")
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument(
        "--device",
        default="0",
        type=int,
        help="Running device. E.g `--device 0`, if using cpu, set `--device -1`",
    )
    parser.add_argument("--save_dir", type=str, help="The saving directory")
    parser.add_argument("--save_id", type=int, help="The saving log id")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument(
        "--data_name",
        default="ml-1m",
        type=str,
        help="The dataset name: ml-100k, ml-1m, ml-10m",
    )
    parser.add_argument(
        "--data_test_ratio", type=float, default=0.1
    )  ## for ml-100k the test ration is 0.2
    parser.add_argument("--data_valid_ratio", type=float, default=0.1)
    parser.add_argument("--use_one_hot_fea", action="store_true", default=False)
    parser.add_argument("--model_activation", type=str, default="leaky")
    parser.add_argument("--gcn_dropout", type=float, default=0.7)
    parser.add_argument("--gcn_agg_norm_symm", type=bool, default=True)
    parser.add_argument("--gcn_agg_units", type=int, default=500)
    parser.add_argument("--gcn_agg_accum", type=str, default="sum")
    parser.add_argument("--gcn_out_units", type=int, default=75)
    parser.add_argument("--gen_r_num_basis_func", type=int, default=2)
    parser.add_argument("--train_max_iter", type=int, default=2000)
    parser.add_argument("--train_log_interval", type=int, default=1)
    parser.add_argument("--train_valid_interval", type=int, default=1)
    parser.add_argument("--train_optimizer", type=str, default="adam")
    parser.add_argument("--train_grad_clip", type=float, default=1.0)
    parser.add_argument("--train_lr", type=float, default=0.01)
    parser.add_argument("--train_min_lr", type=float, default=0.001)
    parser.add_argument("--train_lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--train_decay_patience", type=int, default=50)
    parser.add_argument(
        "--train_early_stopping_patience", type=int, default=100
    )
    parser.add_argument("--share_param", default=False, action="store_true")

    args = parser.parse_args()
    args.device = (
        th.device(args.device) if args.device >= 0 else th.device("cpu")
    )

    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = (
            args.data_name
            + "_"
            + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=2)
            )
        )
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == "__main__":
    args = config()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)
