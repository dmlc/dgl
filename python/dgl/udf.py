"""User-defined function related data structures."""
from __future__ import absolute_import

from collections import Mapping

from .base import ALL, is_all
from . import backend as F
from . import utils

class EdgeBatch(object):
    """The object that represents a batch of edges.

    Parameters
    ----------
    g : DGLGraph
        The graph object.
    edges : tuple of utils.Index
        The edge tuple (u, v, eid). eid can be ALL
    src_data : dict of tensors
        The src node features
    edge_data : dict of tensors
        The edge features.
    dst_data : dict of tensors
        The dst node features
    """
    def __init__(self, g, edges, src_data, edge_data, dst_data):
        self._g = g
        self._edges = edges
        self._src_data = src_data
        self._edge_data = edge_data
        self._dst_data = dst_data

    @property
    def src(self):
        """Return the feature data of the source nodes.

        Returns
        -------
        dict of str to tensors
            The feature data.
        """
        return self._src_data

    @property
    def dst(self):
        """Return the feature data of the destination nodes.

        Returns
        -------
        dict of str to tensors
            The feature data.
        """
        return self._dst_data

    @property
    def data(self):
        """Return the edge feature data.

        Returns
        -------
        dict of str to tensors
            The feature data.
        """
        return self._edge_data

    def edges(self):
        """Return the edges contained in this batch.
        
        Returns
        -------
        tuple of tensors
            The edge tuple (u, v, eid).
        """
        if is_all(self._edges[2]):
            self._edges[2] = utils.toindex(F.arange(
                0, self._g.number_of_edges(), dtype=F.int64))
        u, v, eid = self._edges
        return (u.tousertensor(), v.tousertensor(), eid.tousertensor())

    def batch_size(self):
        """Return the number of edges in this edge batch."""
        return len(self._edges[0])

    def __len__(self):
        """Return the number of edges in this edge batch."""
        return self.batch_size()

class NodeBatch(object):
    """The object that represents a batch of nodes.

    Parameters
    ----------
    g : DGLGraph
        The graph object.
    nodes : utils.Index or ALL
        The node ids.
    data : dict of tensors
        The node features
    msgs : dict of tensors, optional
        The messages.
    """
    def __init__(self, g, nodes, data, msgs=None):
        self._g = g
        self._nodes = nodes
        self._data = data
        self._msgs = msgs

    @property
    def data(self):
        """Return the node feature data.

        Returns
        -------
        dict of str to tensors
            The feature data.
        """
        return self._data

    @property
    def mailbox(self):
        """Return the received messages.

        If no messages received, a None will be returned.

        Returns
        -------
        dict of str to tensors
            The message data.
        """
        return self._msgs

    def nodes(self):
        """Return the nodes contained in this batch.
        
        Returns
        -------
        tensor
            The nodes.
        """
        if is_all(self._nodes):
            self._nodes = utils.toindex(F.arange(
                0, self._g.number_of_nodes(), dtype=F.int64))
        return self._nodes.tousertensor()

    def batch_size(self):
        """Return the number of nodes in this node batch."""
        if is_all(self._nodes):
            return self._g.number_of_nodes()
        else:
            return len(self._nodes)

    def __len__(self):
        """Return the number of nodes in this node batch."""
        return self.batch_size()
