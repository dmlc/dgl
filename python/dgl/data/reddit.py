from __future__ import absolute_import

import scipy.sparse as sp
import numpy as np
import dgl
import os, sys
from ..graph_index import create_graph_index
from .utils import download, extract_archive, get_download_dir, _get_dgl_url


class RedditDataset(object):
    def __init__(self, self_loop=False):
        download_dir = get_download_dir()
        self_loop_str = ""
        if self_loop:
            self_loop_str = "_self_loop"
        zip_file_path = os.path.join(download_dir, "reddit{}.zip".format(self_loop_str))
        download(_get_dgl_url("dataset/reddit{}.zip".format(self_loop_str)), path=zip_file_path)
        extract_dir = os.path.join(download_dir, "reddit{}".format(self_loop_str))
        extract_archive(zip_file_path, extract_dir)
        # graph
        coo_adj = sp.load_npz(os.path.join(extract_dir, "reddit{}_graph.npz".format(self_loop_str)))
        self.graph = create_graph_index(coo_adj, readonly=True)
        # features and labels
        reddit_data = np.load(os.path.join(extract_dir, "reddit_data.npz"))
        self.features = reddit_data["feature"]
        self.labels = reddit_data["label"]
        self.num_labels = 41
        # tarin/val/test indices
        node_ids = reddit_data["node_ids"]
        node_types = reddit_data["node_types"]
        self.train_mask = (node_types == 1)
        self.val_mask = (node_types == 2)
        self.test_mask = (node_types == 3)

