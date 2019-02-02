from __future__ import absolute_import

import scipy.sparse as sp
import numpy as np
import dgl
import os, sys
from .utils import download, extract_archive, get_download_dir, _get_dgl_url


class RedditDataset(object):
    def __init__(self):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(download_dir, "reddit.zip")
        download(_get_dgl_url("dataset/reddit.zip"), path=zip_file_path)
        extract_dir = os.path.join(download_dir, "reddit")
        extract_archive(zip_file_path, extract_dir)
        # graph
        coo_adj = sp.load_npz(os.path.join(extract_dir, "reddit_graph.npz"))
        self.graph = dgl.DGLGraph(coo_adj, readonly=True)
        # features and labels
        reddit_data = np.load(os.path.join(extract_dir, "reddit_data.npz"))
        self.features = reddit_data["feature"]
        self.labels = reddit_data["label"]
        # tarin/val/test indices
        node_ids = reddit_data["node_ids"]
        node_types = reddit_data["node_types"]
        self.train_indices = node_ids[node_types == 1]
        self.val_indices = node_ids[node_types == 2]
        self.test_indices = node_ids[node_types == 3]

