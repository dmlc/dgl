"""KarateClub Dataset
"""
import numpy as np
import networkx as nx

from .. import backend as F
from .dgl_dataset import DGLDataset
from .utils import deprecate_property
from ..convert import graph as dgl_graph

__all__ = ['KarateClubDataset', 'KarateClub']


class KarateClubDataset(DGLDataset):
    r"""

    Description
    -----------
    Zachary's karate club is a social network of a university
    karate club, described in the paper "An Information Flow
    Model for Conflict and Fission in Small Groups" by Wayne W. Zachary.
    The network became a popular example of community structure in
    networks after its use by Michelle Girvan and Mark Newman in 2002.
    Official website: http://konect.cc/networks/ucidata-zachary/

    Statistics
    ----------
    Nodes: 34
    Edges: 156
    Number of Classes: 2

    Attributes
    ----------
    num_classes : int
        Number of node classes

    Returns
    -------
    KarateClubDataset

    Examples
    --------
    >>> data = KarateClubDataset()
    >>> data.num_classes
    2
    >>> g = data[0]
    >>> g.ndata
    {'label': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
    """
    def __init__(self):
        super(KarateClubDataset, self).__init__(name='karate_club')

    def process(self):
        kc_graph = nx.karate_club_graph()
        label = np.asarray(
            [kc_graph.nodes[i]['club'] != 'Mr. Hi' for i in kc_graph.nodes]).astype(np.int64)
        label = F.tensor(label)
        g = dgl_graph(kc_graph)
        g.ndata['label'] = label
        self._graph = g
        self._data = [g]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    @property
    def data(self):
        deprecate_property('dataset.data', 'dataset[0]')
        return self._data

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._graph

    def __len__(self):
        return 1


KarateClub = KarateClubDataset
