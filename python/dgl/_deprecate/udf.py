"""User-defined function related data structures."""
from __future__ import absolute_import

class EdgeBatch(object):
    """The class that can represent a batch of edges.

    Parameters
    ----------
    edges : tuple of utils.Index
        The edge tuple (u, v, eid). eid can be ALL
    src_data : dict
        The src node features, in the form of ``dict``
        with ``str`` keys and ``tensor`` values
    edge_data : dict
        The edge features, in the form of ``dict`` with
        ``str`` keys and ``tensor`` values
    dst_data : dict of tensors
        The dst node features, in the form of ``dict``
        with ``str`` keys and ``tensor`` values
    canonical_etype : tuple of (str, str, str), optional
        Canonical edge type of the edge batch, if UDF is
        running on a heterograph.
    """
    def __init__(self, edges, src_data, edge_data, dst_data,
                 canonical_etype=(None, None, None)):
        self._edges = edges
        self._src_data = src_data
        self._edge_data = edge_data
        self._dst_data = dst_data
        self._canonical_etype = canonical_etype

    @property
    def src(self):
        """Return the feature data of the source nodes.

        Returns
        -------
        dict with str keys and tensor values
            Features of the source nodes.
        """
        return self._src_data

    @property
    def dst(self):
        """Return the feature data of the destination nodes.

        Returns
        -------
        dict with str keys and tensor values
            Features of the destination nodes.
        """
        return self._dst_data

    @property
    def data(self):
        """Return the edge feature data.

        Returns
        -------
        dict with str keys and tensor values
            Features of the edges.
        """
        return self._edge_data

    def edges(self):
        """Return the edges contained in this batch.

        Returns
        -------
        tuple of three tensors
            The edge tuple :math:`(src, dst, eid)`. :math:`src[i],
            dst[i], eid[i]` separately specifies the source node,
            destination node and the edge id for the ith edge
            in the batch.
        """
        u, v, eid = self._edges
        return (u.tousertensor(), v.tousertensor(), eid.tousertensor())

    def batch_size(self):
        """Return the number of edges in this edge batch.

        Returns
        -------
        int
        """
        return len(self._edges[0])

    def __len__(self):
        """Return the number of edges in this edge batch.

        Returns
        -------
        int
        """
        return self.batch_size()

    @property
    def canonical_etype(self):
        """Return the canonical edge type (i.e. triplet of source, edge, and
        destination node type) for this edge batch, if available."""
        return self._canonical_etype

class NodeBatch(object):
    """The class that can represent a batch of nodes.

    Parameters
    ----------
    nodes : utils.Index
        The node ids.
    data : dict
        The node features, in the form of ``dict``
        with ``str`` keys and ``tensor`` values
    msgs : dict, optional
        The messages, , in the form of ``dict``
        with ``str`` keys and ``tensor`` values
    ntype : str, optional
        The node type of this node batch, if running
        on a heterograph.
    """
    def __init__(self, nodes, data, msgs=None, ntype=None):
        self._nodes = nodes
        self._data = data
        self._msgs = msgs
        self._ntype = ntype

    @property
    def data(self):
        """Return the node feature data.

        Returns
        -------
        dict with str keys and tensor values
            Features of the nodes.
        """
        return self._data

    @property
    def mailbox(self):
        """Return the received messages.

        If no messages received, a ``None`` will be returned.

        Returns
        -------
        dict or None
            The messages nodes received. If dict, the keys are
            ``str`` and the values are ``tensor``.
        """
        return self._msgs

    def nodes(self):
        """Return the nodes contained in this batch.

        Returns
        -------
        tensor
            The nodes.
        """
        return self._nodes.tousertensor()

    def batch_size(self):
        """Return the number of nodes in this batch.

        Returns
        -------
        int
        """
        return len(self._nodes)

    def __len__(self):
        """Return the number of nodes in this node batch.

        Returns
        -------
        int
        """
        return self.batch_size()

    @property
    def ntype(self):
        """Return the node type of this node batch, if available."""
        return self._ntype
