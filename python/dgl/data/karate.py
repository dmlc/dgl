"""KarateClub Dataset
"""
import numpy as np
import networkx as nx

from .dgl_dataset import DGLDataset
from ..graph import DGLGraph
from ..base import dgl_warning

__all__ = ['KarateClubDataset', 'KarateClub']


class KarateClubDataset(DGLDataset):
    """KarateClub dataset.

    Zachary's karate club is a social network of a university
    karate club, described in the paper "An Information Flow
    Model for Conflict and Fission in Small Groups" by Wayne W. Zachary.
    The network became a popular example of community structure in
    networks after its use by Michelle Girvan and Mark Newman in 2002.
    Official website: http://konect.cc/networks/ucidata-zachary/

    Statistics
    ===
    Nodes: 34
    Edges: 156
    Number of Classes: 2

    Returns
    ===
    KarateClubDataset object with two properties:
        graph: A Homogeneous graph contains the graph structure and node labels
        num_classes: number of node classes

    Examples
    ===
    >>> data = KarateClubDataset()
    >>> data.num_classes
    2
    >>> g = data.graph
    >>> g.ndata
    {'label': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
    """
    def __init__(self):
        super(KarateClubDataset, self).__init__(name='karate_club')

    def process(self, root_path):
        kc_graph = nx.karate_club_graph()
        self.label = np.asarray(
            [kc_graph.nodes[i]['club'] != 'Mr. Hi' for i in kc_graph.nodes]).astype(np.int64)
        g = DGLGraph(kc_graph)
        g.ndata['label'] = self.label
        self.graph = g
        self.data = [g]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        return 1


class KarateClub(KarateClubDataset):
    def __init__(self):
        dgl_warning('KarateClub is deprecated, use KarateClubDataset instead.',
                    DeprecationWarning, stacklevel=2)
        super(KarateClub, self).__init__()

