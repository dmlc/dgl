"""User-defined function related data structures."""
from __future__ import absolute_import

class EdgeBatch(object):
    """The class that can represent a batch of edges.

    Parameters
    ----------
    graph : DGLGraph
        Graph object.
    eid : Tensor
        Edge IDs.
    etype : (str, str, str)
        Edge type.
    src_data : dict[str, Tensor]
        Src node features.
    edge_data : dict[str, Tensor]
        Edge features.
    dst_data : dict[str, Tensor]
        Dst node features.
    """
    def __init__(self, graph, eid, etype, src_data, edge_data, dst_data):
        self._graph = graph
        self._eid = eid
        self._etype = etype
        self._src_data = src_data
        self._edge_data = edge_data
        self._dst_data = dst_data

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
        Tensor
            Source node IDs.
        Tensor
            Destination node IDs.
        Tensor
            Edge IDs.
        """
        u, v = self._graph.find_edges(self._eid, etype=self.canonical_etype)
        return u, v, self._eid

    def batch_size(self):
        """Return the number of edges in this edge batch.

        Returns
        -------
        int
        """
        return len(self._eid)

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
        destination node type) for this edge batch."""
        return self._etype

class NodeBatch(object):
    """The class to represent a batch of nodes.

    Parameters
    ----------
    graph : DGLGraph
        Graph object.
    nodes : Tensor
        Node ids.
    ntype : str, optional
        The node type of this node batch,
    data : dict[str, Tensor]
        Node feature data.
    msgs : dict[str, Tensor], optional
        Messages data.
    """
    def __init__(self, graph, nodes, ntype, data, msgs=None):
        self._graph = graph
        self._nodes = nodes
        self._ntype = ntype
        self._data = data
        self._msgs = msgs

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
        return self._nodes

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
