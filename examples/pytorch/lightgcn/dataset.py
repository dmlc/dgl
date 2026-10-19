"""
Dataset module for LightGCN.
Handles data loading and graph construction.
"""
import dgl
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from config import args
from utils import *

class Dataset:
    def __init__(self, path):
        self.path = path
        self.device = torch.device(args.device)
        self.train_file = path + "/train.txt"
        self.test_file = path + "/test.txt"
        self._load_data()
        self._build_graph()
        test_dict = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_dict.get(user):
                test_dict[user].append(item)
            else:
                test_dict[user] = [item]
        self.test_dict = test_dict
        self.n_edges = self.hetero_graph.num_edges(etype='interacts')

    def _load_data(self):
        train_user, train_item = [], []
        test_user, test_item = [], []
        allPos = {}
        with open(self.train_file) as f:
            for line in f:
                l = line.strip().split()
                u = int(l[0])
                items = [int(i) for i in l[1:]]
                train_user.extend([u] * len(items))
                train_item.extend(items)
                allPos.setdefault(u, []).extend(items)
        with open(self.test_file) as f:
            for line in f:
                l = line.strip().split()
                u = int(l[0])
                items = [int(i) for i in l[1:]]
                test_user.extend([u] * len(items))
                test_item.extend(items)

        self.train_user, self.train_item = np.array(train_user), np.array(train_item)
        self.test_user, self.test_item = np.array(test_user), np.array(test_item)
        self.n_users = self.train_user.max() + 1
        self.n_items = self.train_item.max() + 1
        self.allPos = allPos
        self.allPos_sets = {user: set(items) for user, items in allPos.items()}
    def _build_graph(self):
        # user → item
        src = torch.from_numpy(self.train_user)
        dst = torch.from_numpy(self.train_item)
        self.hetero_graph = dgl.heterograph({
            ('user', 'interacts', 'item'): (src, dst),
        })
        # Create bidirectional bipartite graph
        # User nodes：0 to n_users-1
        # Item nodes：n_users to n_users+n_items-1
        dst_offset = dst + self.n_users
        # Create edges：user->item and item->user
        src_bidir = torch.cat([src, dst_offset])
        dst_bidir = torch.cat([dst_offset, src])
        self.graph = dgl.graph((src_bidir, dst_bidir), num_nodes=self.n_users + self.n_items)
        self.heterograph = self.hetero_graph.to(self.device)
        self.graph = self.graph.to(self.device)
        print("graph for training:", self.hetero_graph)

    def build_train_dataset(self):
        users, pos_i, neg_i = sample_triplets(self.hetero_graph, self.allPos, self.allPos_sets, self.hetero_graph.num_edges(etype='interacts'))
        users = users.to(self.device)
        pos_i = pos_i.to(self.device)
        neg_i = neg_i.to(self.device)
        users, pos_i, neg_i = shuffle(users, pos_i, neg_i)
        # users = torch.load("/home/xty/dgl-ascend/examples/pytorch/lightgcn/debug_user.pt", map_location="cpu")
        # pos_i = torch.load("/home/xty/dgl-ascend/examples/pytorch/lightgcn/debug_pos_items.pt", map_location="cpu")
        # neg_i = torch.load("/home/xty/dgl-ascend/examples/pytorch/lightgcn/debug_neg_items.pt", map_location="cpu")
        self.train_dataset = TensorDataset(users, pos_i, neg_i)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch, shuffle=False)
        return self.train_loader
