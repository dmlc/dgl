"""KarateClub Dataset
"""
import numpy as np
import networkx as nx
from dgl import DGLGraph


class KarateClub(object):
    """
    Zachary's karate club is a social network of a university karate club, described in the paper
    "An Information Flow Model for Conflict and Fission in Small Groups" by Wayne W. Zachary. The
    network became a popular example of community structure in networks after its use by Michelle
    Girvan and Mark Newman in 2002.

    This dataset has only one graph, with ndata 'label' means whether the node is belong to the "Mr. Hi" club.
    """

    def __init__(self):
        kG = nx.karate_club_graph()
        self.label = np.array(
            [kG.node[i]['club'] != 'Mr. Hi' for i in kG.nodes]).astype(np.int64)
        g = DGLGraph(kG)
        g.ndata['label'] = self.label
        self.data = [g]

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.data[0]

    def __len__(self):
        return len(self.data)
