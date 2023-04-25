# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/load_data.py>.
# It implements the data processing and graph construction.
import random as rd

import dgl

import numpy as np


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + "/train.txt"
        test_file = path + "/test.txt"

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.exist_users = []

        user_item_src = []
        user_item_dst = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split(" ")
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    for i in l[1:]:
                        user_item_src.append(uid)
                        user_item_dst.append(int(i))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n")
                    try:
                        items = [int(i) for i in l.split(" ")[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        # training positive items corresponding to each user; testing positive items corresponding to each user
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip("\n")
                    items = [int(i) for i in l.split(" ")]
                    uid, train_items = items[0], items[1:]
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip("\n")
                    try:
                        items = [int(i) for i in l.split(" ")]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        # construct graph from the train data and add self-loops
        user_selfs = [i for i in range(self.n_users)]
        item_selfs = [i for i in range(self.n_items)]

        data_dict = {
            ("user", "user_self", "user"): (user_selfs, user_selfs),
            ("item", "item_self", "item"): (item_selfs, item_selfs),
            ("user", "ui", "item"): (user_item_src, user_item_dst),
            ("item", "iu", "user"): (user_item_dst, user_item_src),
        }
        num_dict = {"user": self.n_users, "item": self.n_items}

        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [
                rd.choice(self.exist_users) for _ in range(self.batch_size)
            ]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if (
                    neg_id not in self.train_items[u]
                    and neg_id not in neg_items
                ):
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_test))
        print(
            "n_train=%d, n_test=%d, sparsity=%.5f"
            % (
                self.n_train,
                self.n_test,
                (self.n_train + self.n_test) / (self.n_users * self.n_items),
            )
        )
