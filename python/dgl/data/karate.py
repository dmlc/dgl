"""KarateClub Dataset
"""
import numpy as np
import networkx as nx

from .. import backend as F
from .dgl_dataset import DGLDataset
from .utils import deprecate_property
from ..convert import from_networkx

__all__ = ['KarateClubDataset', 'KarateClub']


class KarateClubDataset(DGLDataset):
    r""" Karate Club dataset for Node Classification

    .. deprecated:: 0.5.0

        - ``data`` is deprecated, it is replaced by:

            >>> dataset = KarateClubDataset()
            >>> g = dataset[0]

    Zachary's karate club is a social network of a university
    karate club, described in the paper "An Information Flow
    Model for Conflict and Fission in Small Groups" by Wayne W. Zachary.
    The network became a popular example of community structure in
    networks after its use by Michelle Girvan and Mark Newman in 2002.
    Official website: `<http://konect.cc/networks/ucidata-zachary/>`_

    Karate Club dataset statistics:

    - Nodes: 34
    - Edges: 156
    - Number of Classes: 2

    Attributes
    ----------
    num_classes : int
        Number of node classes
    data : list
        A list of :class:`dgl.DGLGraph` objects

    Examples
    --------
    >>> dataset = KarateClubDataset()
    >>> num_classes = dataset.num_classes
    >>> g = dataset[0]
    >>> labels = g.ndata['label']
    """
    def __init__(self):
        super(KarateClubDataset, self).__init__(name='karate_club')

    def process(self):
        kc_graph = nx.karate_club_graph()
        label = np.asarray(
            [kc_graph.nodes[i]['club'] != 'Mr. Hi' for i in kc_graph.nodes]).astype(np.int64)
        label = F.tensor(label)
        g = from_networkx(kc_graph)
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
        r""" Get graph object

        Parameters
        ----------
        idx : int
            Item index, KarateClubDataset has only one graph object

        Returns
        -------
        :class:`dgl.DGLGraph`

            graph structure and labels.

            - ``ndata['label']``: ground truth labels
        """
        assert idx == 0, "This dataset has only one graph"
        return self._graph

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1


KarateClub = KarateClubDataset
