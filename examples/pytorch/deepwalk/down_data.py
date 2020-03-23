import torch
import argparse
import csv
import multiprocessing as mp
import os
import random
import torch.optim as optim
#from torch.utils.data import DataLoader

from tqdm import tqdm

from reading_data import DeepwalkDataset
from model import SkipGramModel

def ReadTxtNet(file_path=""):
    net = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            n1, n2 = list(map(int, line.strip().split(" ")[:2]))
            try:
                net[n1][n2] = 1
            except:
                net[n1] = {n2: 1}
            try:
                net[n2][n1] = 1
            except:
                net[n2] = {n1: 1}
    print("node num: %d" % len(net))
    print("edge num: %d" % (sum(list(map(lambda i: len(net[i]), net.keys())))/2))
    if max(net.keys()) != len(net) - 1:
        print("error reading net, quit")
        exit(1)
    return net

def ReadCSVNet(file_path=""):
    net = {}
    with open(file_path, "r") as f:
        fcsv = csv.reader(f)
        for row in fcsv:
            n1, n2 = list(map(int, row))
            n1 -= 1
            n2 -= 1
            try:
                net[n1][n2] = 1
            except:
                net[n1] = {n2: 1}
            try:
                net[n2][n1] = 1
            except:
                net[n2] = {n1: 1}
    print("node num: %d" % len(net))
    print("edge num: %d" % (sum(list(map(lambda i: len(net[i]), net.keys())))/2))
    if max(net.keys()) != len(net) - 1:
        print("error reading net, quit")
        exit(1)
    return net

def sampler(all_walks, neg_table, args):
    batch_id = 0
    # batch size = 128
    batch_num = len(all_walks)/args.batch_size
    print("batch size: %d; batch num: %d" % (args.batch_size, batch_num))
    with open(args.output_file, "w") as f:
        while True:
            walk_id = batch_id * args.batch_size
            if batch_id * args.batch_size >= len(all_walks):
                return
            else:
                walks = all_walks[walk_id: walk_id + args.batch_size]

            if batch_id > 0 and batch_id % 50 == 0:
                print("Batch %d" % batch_id)
    
            batch_id += 1

            # solution 1
            for pidx in range(1, args.walk_length-1):
                pos_v = []
                pos_u = []
                neg_v = []
                for cidx in list(range(max(0, pidx-args.window_size),pidx)) +\
                        list(range(pidx+1, min(args.walk_length, pidx+1+args.window_size))):
                    for walk in walks:
                        pos_v.append(walk[pidx])
                        pos_u.append(walk[cidx])
                        neg_v.append(random.sample(neg_table, args.negative))
                for u, v, nv in zip(pos_u, pos_v, neg_v):
                    f.write("%s\n" % " ".join(list(map(str, [u] + [v] + nv))))
            # solution 2
#            rand = random.Random()
#            for walk in walks:
#                for pidx in range(1, 
                


class DeepwalkTrainer:
    '''
    train with negative sampling
    '''
    def __init__(self, args):

        self.dataset = DeepwalkDataset(ReadTxtNet(args.net), args)
        self.output_file_name = args.output_file
        self.emb_size = len(self.dataset.net)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr

        walks = self.dataset.walks
        sampler(walks, self.dataset.neg_table, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Metapath2vec")
    parser.add_argument('--net', type=str, help="download_path")
    parser.add_argument('--output_file', type=str, help='output_file')
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=10, type=int, help="context window size")
    parser.add_argument('--num_walks', default=10, type=int, help="context window size")
    parser.add_argument('--negative', default=5, type=int, help="negative samples")
    parser.add_argument('--iterations', default=1, type=int, help="iterations")
    parser.add_argument('--batch_size', default=10, type=int, help="number of node sequences in each step")
    parser.add_argument('--walk_length', default=80, type=int, help="")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")
    args = parser.parse_args()
    model = DeepwalkTrainer(args)
