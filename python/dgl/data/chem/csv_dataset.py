from __future__ import absolute_import

import dgl.backend as F
import numpy as np
import os
import pickle
import sys

from dgl import DGLGraph
from .utils import smile2graph
from ..utils import download, get_download_dir, _get_dgl_url, Subset

class CSVDataset(object):
    
    def __init__(self, df, smile2graph=smile2graph, smile_name='smiles', cache_file_path="csvdata_dglgraph.pkl"):
        if 'rdkit' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning("Please install RDKit (Recommended Version is 2018.09.3)")
        self.df = df
        self.smiles = self.df[smile_name].tolist()
        self.task_names = self.df.columns.drop([smile_name]).tolist()
        self.cache_file_path = cache_file_path
        self._pre_process(smile2graph)

    def _pre_process(self, smile2graph):
        if os.path.exists(self.cache_file_path):
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
        else:
            self.graphs = []
            for id, s in enumerate(self.smiles):
                self.graphs.append(smile2graph(s))

            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)

        _label_values = self.df[self.task_names].values
        self.labels = np.nan_to_num(_label_values)
        self.mask = ~np.isnan(_label_values)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

    def __len__(self):
        return len(self.smiles)
