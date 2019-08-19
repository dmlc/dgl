from __future__ import absolute_import

import dgl.backend as F
import numpy as np
import os
import pickle
import sys

from dgl import DGLGraph
from .utils import smile2graph


try:
    import pandas as pd
    import rdkit
except ImportError:
    pass

class CSVDataset(object):
    """CSVDataset

    This is a general class for loading data from csv or pd.DataFrame.

    In data pre-processing, we set non-existing labels to be 0,
    and returning mask with 1 where label exists.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs will be saved for reloading so that we do not need to reconstruct them every time.

    Parameters
    ----------
    df: pandas.DataFrame
    Dataframe including smiles and labels. Can be loaded by pandas.read_csv(file_path).
    One column includes smiles and other columns for labels.
    Column names other than smiles column would be considered as task names.

    smile2graph: callable, str -> DGLGraph
    A function turns smiles into a DGLGraph. Default one can be found 
    at python/dgl/data/chem/utils.py named with smile2graph.

    smile_column: str
    Column name that including smiles

    cache_file_path: str
    Path to store the preprocessed data
    """

    def __init__(self, csvfile_path, smile2graph=smile2graph, smile_column='smiles', id_column=None,
                 cache_file_path="csvdata_dglgraph.pkl"):
        if 'rdkit' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning(
                "Please install RDKit (Recommended Version is 2018.09.3)")
        if 'pandas' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning("Please install pandas")
        self.df = pd.read_csv(csvfile_path)
        if id_column is not None:
            self.id = self.df[id_column]
            self.df = self.df.drop(columns=[id_column])
        self.smiles = self.df[smile_column].tolist()
        self.task_names = self.df.columns.drop([smile_column]).tolist()
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(smile2graph)

    def _pre_process(self, smile2graph):
        """Pre-process the dataset

        * Convert molecules from smiles format into DGLGraphs
          and featurize their atoms
        * Set missing labels to be 0 and use a binary masking
          matrix to mask them
        """
        if os.path.exists(self.cache_file_path):
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
        else:
            self.graphs = [smile2graph(s) for s in self.smiles]
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)

        _label_values = self.df[self.task_names].values
        # np.nan_to_num will also turn inf into a very large number
        self.labels = np.nan_to_num(_label_values).astype(np.float32)
        self.mask = (~np.isnan(_label_values)).astype(np.float32)

    def __getitem__(self, item):
        """Get the ith datapoint

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        Tensor of dtype float32
            Weights of the datapoint for all tasks
        """
        return self.smiles[item], self.graphs[item], \
            F.zerocopy_from_numpy(self.labels[item]),  \
            F.zerocopy_from_numpy(self.mask[item])

    def __len__(self):
        """Length of Dataset

        Return
        ------
        int
            Length of Dataset
        """
        return len(self.smiles)
