from __future__ import absolute_import

import scipy.sparse as sp
import numpy as np
import dgl
import os, sys
from .utils import download, get_download_dir, _get_dgl_url


class RedditDataset(object):
    def __init__(self):
        download_dir = get_download_dir()
        # TODO upload the .npz files to S3
        # graph
        coo_adj = sp.load_npz(os.path.join(download_dir, "reddit_graph.npz"))
        self.graph = dgl.DGLGraph(coo_adj, readonly=True)
        # features and labels
        reddit_data = np.load(os.path.join(download_dir, "reddit_data.npz"))
        self.features = reddit_data['feature']
        self.labels = reddit_data['label']
        # tarin/val/test indices
        node_ids = reddit_data["node_ids"]
        node_types = reddit_data["node_types"]
        self.train_indices = node_ids[node_types == 1]
        self.val_indices = node_ids[node_types == 2]
        self.test_indices = node_ids[node_types == 3]

