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
        """Return a view of the source node features for the edges in the batch.

        Examples
        --------

        >>> # For an EdgeBatch instance, get the feature 'h' for all source nodes.
        >>> # The feature is a tensor of shape (E, *),
        >>> # where E is the number of edges in the batch.
        >>> edges.src['h']
        """
        return self._src_data

    @property
    def dst(self):
        """Return a view of the destination node features for the edges in the batch.

        Examples
        --------

        >>> # For an EdgeBatch instance, get the feature 'h' for all destination nodes.
        >>> # The feature is a tensor of shape (E, *),
        >>> # where E is the number of edges in the batch.
        >>> edges.dst['h']
        """
        return self._dst_data

    @property
    def data(self):
        """Return a view of the edge features for the edges in the batch.

        Examples
        --------

        >>> # For an EdgeBatch instance, get the feature 'h' for all edges
        >>> # The feature is a tensor of shape (E, *),
        >>> # where E is the number of edges in the batch.
        >>> edges.data['h']
        """
        return self._edge_data

    def edges(self):
        """Return the edges in the batch

        Returns
        -------
        (U, V, EID) : (Tensor, Tensor, Tensor)
            The edges in the batch. For each :math:`i`, :math:`(U[i], V[i])` is an edge
            from :math:`U[i]` to :math:`V[i]` with ID :math:`EID[i]`.
        """
        u, v = self._graph.find_edges(self._eid, etype=self.canonical_etype)
        return u, v, self._eid

    def batch_size(self):
        """Return the number of edges in the batch.

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
        """Return a view of the node features for the nodes in the batch.

        Examples
        --------

        >>> # For a NodeBatch instance, get the feature 'h' for all nodes
        >>> # The feature is a tensor of shape (N, *),
        >>> # where N is the number of nodes in the batch.
        >>> nodes.data['h']
        """
        return self._data

    @property
    def mailbox(self):
        """Return a view of the messages received.

        Examples
        --------

        >>> # For a NodeBatch instance, get the messages 'm' for all nodes.
        >>> # The messages is a tensor of shape (N, M, *), where N is the
        >>> # number of nodes in the batch and M is the number of messages
        >>> # each node receives.
        >>> nodes.mailbox['m']
        """
        return self._msgs

    def nodes(self):
        """Return the nodes in the batch.

        Returns
        -------
        NID : Tensor
            The IDs of the nodes in the batch. :math:`NID[i]` gives the ID of
            the i-th node.
        """
        return self._nodes

    def batch_size(self):
        """Return the number of nodes in the batch.

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
