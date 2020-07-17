"""ICEWS18 dataset for temporal graph"""
import numpy as np
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import loadtxt, _get_dgl_url, save_graphs, load_graphs
from ..graph import DGLGraph


class ICEWS18Dataset(DGLBuiltinDataset):
    r""" ICEWS18 dataset for temporal graph

    Integrated Crisis Early Warning System (ICEWS18)
    Event data consists of coded interactions between socio-political
    actors (i.e., cooperative or hostile actions between individuals,
    groups, sectors and nation states).
    This Dataset consists of events from 1/1/2018 to 10/31/2018 (24 hours time granularity).

    Reference:
        - `Recurrent Event Network for Reasoning over Temporal
        Knowledge Graphs <https://arxiv.org/abs/1904.05530>`_
        - `ICEWS Coded Event Data
           <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075>`_

    Statistics
    ----------
    Train examples: 240
    Valid examples: 30
    Test examples: 34
    Nodes per graph: 23033

    Parameters
    ----------
    mode: str
        Load train/valid/test data. Has to be one of ['train', 'valid', 'test']
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Returns
    -------
    ICEWS18Dataset object with one properity
        graphs: list of DGLGraph objects each contains graph structure and edge features
            - edata['rel_type']: relation type
    Examples
    --------
    >>> # get train, valid, test set
    >>> train_data = ICEWS18Dataset()
    >>> valid_data = ICEWS18Dataset(mode='valid')
    >>> test_data = ICEWS18Dataset(mode='test')
    >>>
    >>> train_size = len(train_data)
    >>> for g in train_data:
    ....    e_feat = g.edata['rel_type']
    ....    # your code here
    ....
    >>>
    """
    def __init__(self, mode='train', raw_dir=None, force_reload=False, verbose=False):
        mode = mode.lower()
        assert mode in ['train', 'valid', 'test'], "Mode not valid"
        self.mode = mode
        _url = _get_dgl_url('dataset/icews18.zip')
        super(ICEWS18Dataset, self).__init__(name='ICEWS18',
                                             url=_url,
                                             raw_dir=raw_dir,
                                             force_reload=force_reload,
                                             verbose=verbose)

    def process(self):
        data = loadtxt(os.path.join(self.save_path, '{}.txt'.format(self.mode)),
                       delimiter='\t').astype(np.int64)
        num_nodes = 23033
        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        time_index = np.floor(data[:, 3] / 24).astype(np.int64)
        start_time = time_index[time_index != -1].min()
        end_time = time_index.max()
        self._graphs = []
        for i in range(start_time, end_time + 1):
            g = DGLGraph()
            g.add_nodes(num_nodes)
            row_mask = time_index <= i
            edges = data[row_mask][:, [0, 2]]
            rate = data[row_mask][:, 1]
            g.add_edges(edges[:, 0], edges[:, 1])
            g.edata['rel_type'] = rate.reshape(-1, 1)
            self._graphs.append(g)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, '{}_dgl_graph.bin'.format(self.mode))
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, '{}_dgl_graph.bin'.format(self.mode))
        save_graphs(graph_path, self.graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, '{}_dgl_graph.bin'.format(self.mode))
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def is_temporal(self):
        return True


ICEWS18 = ICEWS18Dataset
