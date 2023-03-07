import time
import random
import argparse

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl

from dgl.ops import gather_mm
from model import IGMC
from data import MovieLens
from dataset import MovieLensDataset, collate_movielens 

def evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.
    for batch in loader:
        with th.no_grad():
            preds, _ = model(batch[0].to(device))
        labels = batch[1].to(device)
        mse += ((preds - labels) ** 2).sum().item() 
    mse /= len(loader.dataset)
    return np.sqrt(mse)

def train_epoch(model, loss_fn, optimizer, arr_lambda, loader, device, log_interval):
    model.train()

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        preds, arr_loss = model(inputs)
        loss = loss_fn(preds, labels).mean() + arr_lambda * arr_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            print("Iter={}, loss={:.4f}, mse={:.4f}, time={:.4f}".format(
                iter_idx, iter_loss/iter_cnt, iter_mse/iter_cnt, np.average(iter_dur)))
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
    return epoch_loss / len(loader.dataset)

def train(args):
    ### prepare data and set model
    movielens = MovieLens()
    test_dataset = MovieLensDataset(
        movielens.test_rating_pairs, movielens.test_rating_values, movielens.train_graph) 
    train_dataset = MovieLensDataset(
        movielens.train_rating_pairs, movielens.train_rating_values, movielens.train_graph)

    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate_movielens)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=collate_movielens)

    hop = 1
    in_feats = (hop+1)*2
    model = IGMC(in_feats=in_feats, 
            ).to(args.device)
    loss_fn = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=0)
    print("Loading network finished ...\n")
    
    best_epoch = 0
    best_rmse = np.inf
    ### declare the loss information
    print("Start training ...")
    for epoch_idx in range(1, args.train_epochs+1):
        print ('Epoch', epoch_idx)
    
        train_loss = train_epoch(model, loss_fn, optimizer, args.arr_lambda, 
                                train_loader, args.device, 100)
        test_rmse = evaluate(model, test_loader, args.device)
        eval_info = {
            'epoch': epoch_idx,
            'train_loss': train_loss,
            'test_rmse': test_rmse,
        }
        print('=== Epoch {}, train loss {:.6f}, test rmse {:.6f} ==='.format(*eval_info.values()))

        if epoch_idx % args.train_lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.train_lr_decay_factor * param['lr']

        print(f"eval_info: {eval_info}")
        if best_rmse > test_rmse:
            best_rmse = test_rmse
            best_epoch = epoch_idx
    print("Training ends. The best testing rmse is {:.6f} at epoch {}".format(best_rmse, best_epoch))

def config():
    th.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='IGMC')
    # general settings
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--num_workers', type=int, default=8)

    # optimization settings
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--train_lr_decay_step', type=int, default=50)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--train_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--arr_lambda', type=float, default=0.001)
                
    args = parser.parse_args()
    args.device = th.device(args.device) if args.device >= 0 and th.cuda.is_available() else th.device('cpu')
    
    return args

if __name__ == '__main__':
    args = config()
    random.seed(1234)
    np.random.seed(1234)
    th.manual_seed(1234)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(1234)
    dgl.random.seed(1234)
    train(args)
