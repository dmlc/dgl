import torch
import argparse
import dgl
import csv
#import multiprocessing as mp
import os
import random
import time
import numpy as np
#import torch.optim as optim
#from torch.utils.data import DataLoader

from tqdm import tqdm

from reading_data import DeepwalkDataset
from model import SkipGramModel

def fast_train(model, args):
    num_batches = len(model.dataset.net) * args.num_walks / args.batch_size
    num_batches = int(np.ceil(num_batches))
    print("num batchs: %d" % num_batches)

    model.use_cuda = torch.cuda.is_available()
    model.device = torch.device("cuda" if model.use_cuda else "cpu")

    skip_gram_model = SkipGramModel(model.emb_size, 
            model.emb_dimension, model.device, model.mixed_train, 
            args.neg_weight)
    if model.use_cuda and not model.mixed_train:
        print("GPU used")
        skip_gram_model.cuda()
    elif model.use_cuda and model.mixed_train:
        print("Mixed CPU & GPU")
    else:
        print("CPU used")

    def get_onehot(idx, size):
        t = torch.zeros(size)
        t[idx] = 1.
        return t
    
    idx_list = []
    for i in range(args.walk_length):
        for j in range(i-args.window_size, i):
            if j >= 0:
                idx_list.append(j)
        for j in range(i+1, i+1+args.window_size):
            if j < args.walk_length:
                idx_list.append(j)
    if len(idx_list) != int(args.walk_length * args.window_size * 2 - args.window_size * (args.window_size + 1)):
        print("error idx list")
        print(len(idx_list))
        print(args.walk_length * args.window_size * 2 - args.window_size * (args.window_size + 1))
        exit(0)

    # [walk_length, num_item]
    walk2posu = torch.stack([get_onehot(idx, args.walk_length) for idx in idx_list]).to(model.device).T

    idx_list = []
    for i in range(args.walk_length):
        for j in range(i-args.window_size, i):
            if j >= 0:
                idx_list.append(i)
        for j in range(i+1, i+1+args.window_size):
            if j < args.walk_length:
                idx_list.append(i)

    walk2posv = torch.stack([get_onehot(idx, args.walk_length) for idx in idx_list]).to(model.device).T

    def walk2input(walk):
        """ input one sequnce """
        walk = walk.float().to(model.device)
        pos_u = walk.unsqueeze(0).mm(walk2posu).squeeze().long()
        pos_v = walk.unsqueeze(0).mm(walk2posv).squeeze().long()
        neg_u = walk.long()
        t = time.time()
        neg_v = torch.LongTensor(np.random.sample(model.dataset.neg_table, args.negative)).to(model.device)
        return pos_u, pos_v, neg_u, neg_v, time.time()-t

    def walks2input(walks):
        """ input sequences """
        # [batch_size, walk_length]
        bs = len(walks)
        walks = torch.stack(walks).to(model.device).float()
        # [batch_size, num_pos]
        pos_u = walks.mm(walk2posu).long()
        pos_v = walks.mm(walk2posv).long()
        # [batch_size, walk_length]
        neg_u = walks.long()
        t = time.time()
        neg_v = torch.LongTensor(np.random.choice(model.dataset.neg_table, bs * args.negative, replace=True)).to(model.device).view(bs, args.negative)
        return pos_u, pos_v, neg_u, neg_v, time.time()-t

    start_all = time.time()
    start = time.time()
    with torch.no_grad():
        i = 0
        max_i = args.iterations * num_batches
        for iteration in range(model.iterations):
            print("\nIteration: " + str(iteration + 1))
            random.shuffle(model.dataset.walks)

            pre_time = 0.
            sample_time = 0.
            while True:
                lr = args.initial_lr * (max_i - i) / max_i
                if lr < 0.00001:
                    lr = 0.00001

                pre_start = time.time()

                # multi-sequence input
                i_ = int(i % num_batches)
                walks = model.dataset.walks[i_*args.batch_size: \
                        (1+i_)*args.batch_size]
                if len(walks) == 0:
                    break
                pos_u, pos_v, neg_u, neg_v, st = walks2input(walks)

                pre_time += time.time() - pre_start
                sample_time += st

                skip_gram_model.fast_learn_multi(pos_u, pos_v, neg_u, neg_v, lr)

                i += 1
                if i > 0 and i % args.print_interval == 0:
                    print("Batch %d, pt: %.2fs, st: %.2fs, tt: %.2fs" % (i, pre_time, sample_time, time.time()-start))
                    pre_time = 0.
                    sample_time = 0.
                    start = time.time()
                if i_ == num_batches - 1:
                    break

    print("Used time: %.2fs" % (time.time()-start_all))
    skip_gram_model.save_embedding(model.dataset, model.output_file_name)

class DeepwalkTrainer:
    '''
    train with negative sampling
    '''
    def __init__(self, args):

        self.dataset = DeepwalkDataset(args.net_file, args)
        self.output_file_name = args.output_file
        self.emb_size = len(self.dataset.net)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.mixed_train = args.mix

        fast_train(self, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepWalk")
    parser.add_argument('--net_file', type=str, 
            help="network file")
    parser.add_argument('--output_file', type=str, 
            help='embedding file')
    parser.add_argument('--dim', default=128, type=int, 
            help="embedding dimensions")
    parser.add_argument('--window_size', default=5, type=int, 
            help="context window size")
    parser.add_argument('--num_walks', default=10, type=int, 
            help="context window size")
    parser.add_argument('--negative', default=50, type=int, 
            help="negative samples")
    parser.add_argument('--iterations', default=1, type=int, 
            help="iterations")
    parser.add_argument('--batch_size', default=10, type=int, 
            help="number of node sequences in each step")
    parser.add_argument('--print_interval', default=1000, type=int, 
            help="print interval")
    parser.add_argument('--walk_length', default=80, type=int, 
            help="walk length")
    parser.add_argument('--initial_lr', default=0.025, type=float, 
            help="learning rate")
    parser.add_argument('--neg_weight', default=1., type=float, 
            help="negative weight")
    parser.add_argument('--mix', default=False, action="store_true", 
            help="mixed training with CPU and GPU")
    args = parser.parse_args()
    model = DeepwalkTrainer(args)
