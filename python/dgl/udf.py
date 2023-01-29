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
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a node feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that retrieves the source node features for edges.
        >>> def edge_udf(edges):
        >>>     # edges.src['h'] is a tensor of shape (E, 1),
        >>>     # where E is the number of edges in the batch.
        >>>     return {'src': edges.src['h']}

        >>> # Copy features from source nodes to edges.
        >>> g.apply_edges(edge_udf)
        >>> g.edata['src']
        tensor([[1.],
                [1.],
                [1.]])

        >>> # Use edge UDF in message passing, which is equivalent to
        >>> # dgl.function.copy_u.
        >>> import dgl.function as fn
        >>> g.update_all(edge_udf, fn.sum('src', 'h'))
        >>> g.ndata['h']
        tensor([[1.],
                [2.]])
        """
        return self._src_data

    @property
    def dst(self):
        """Return a view of the destination node features for the edges in the batch.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a node feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.tensor([[0.], [1.]])

        >>> # Define a UDF that retrieves the destination node features for
        >>> # edges.
        >>> def edge_udf(edges):
        >>>     # edges.dst['h'] is a tensor of shape (E, 1),
        >>>     # where E is the number of edges in the batch.
        >>>     return {'dst': edges.dst['h']}

        >>> # Copy features from destination nodes to edges.
        >>> g.apply_edges(edge_udf)
        >>> g.edata['dst']
        tensor([[1.],
                [1.],
                [1.]])

        >>> # Use edge UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(edge_udf, fn.sum('dst', 'h'))
        >>> g.ndata['h']
        tensor([[0.],
                [2.]])
        """
        return self._dst_data

    @property
    def data(self):
        """Return a view of the edge features for the edges in the batch.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set an edge feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.edata['h'] = torch.tensor([[1.], [1.], [1.]])

        >>> # Define a UDF that retrieves the feature 'h' for all edges.
        >>> def edge_udf(edges):
        >>>     # edges.data['h'] is a tensor of shape (E, 1),
        >>>     # where E is the number of edges in the batch.
        >>>     return {'data': edges.data['h']}

        >>> # Make a copy of the feature with name 'data'.
        >>> g.apply_edges(edge_udf)
        >>> g.edata['data']
        tensor([[1.],
                [1.],
                [1.]])

        >>> # Use edge UDF in message passing, which is equivalent to
        >>> # dgl.function.copy_e.
        >>> import dgl.function as fn
        >>> g.update_all(edge_udf, fn.sum('data', 'h'))
        >>> g.ndata['h']
        tensor([[1.],
                [2.]])
        """
        return self._edge_data

    def edges(self):
        """Return the edges in the batch.

        Returns
        -------
        (U, V, EID) : (Tensor, Tensor, Tensor)
            The edges in the batch. For each :math:`i`, :math:`(U[i], V[i])` is
            an edge from :math:`U[i]` to :math:`V[i]` with ID :math:`EID[i]`.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))

        >>> # Define a UDF that retrieves and concatenates the end nodes of the
        >>> # edges.
        >>> def edge_udf(edges):
        >>>     src, dst, _ = edges.edges()
        >>>     return {'uv': torch.stack([src, dst], dim=1).float()}

        >>> # Create a feature 'uv' with the end nodes of the edges.
        >>> g.apply_edges(edge_udf)
        >>> g.edata['uv']
        tensor([[0., 1.],
                [1., 1.],
                [1., 0.]])

        >>> # Use edge UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(edge_udf, fn.sum('uv', 'h'))
        >>> g.ndata['h']
        tensor([[1., 0.],
                [1., 2.]])
        """
        u, v = self._graph.find_edges(self._eid, etype=self.canonical_etype)
        return u, v, self._eid

    def batch_size(self):
        """Return the number of edges in the batch.

        Returns
        -------
        int

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))

        >>> # Define a UDF that returns one for each edge.
        >>> def edge_udf(edges):
        >>>     return {'h': torch.ones(edges.batch_size(), 1)}

        >>> # Creates a feature 'h'.
        >>> g.apply_edges(edge_udf)
        >>> g.edata['h']
        tensor([[1.],
                [1.],
                [1.]])

        >>> # Use edge UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(edge_udf, fn.sum('h', 'h'))
        >>> g.ndata['h']
        tensor([[1.],
                [2.]])
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
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received and
        >>> # the original feature for each node.
        >>> def node_udf(nodes):
        >>>     # nodes.data['h'] is a tensor of shape (N, 1),
        >>>     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
        >>>     # where N is the number of nodes in the batch, D is the number
        >>>     # of messages received per node for this node batch.
        >>>     return {'h': nodes.data['h'] + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[2.],
                [3.]])
        """
        return self._data

    @property
    def mailbox(self):
        """Return a view of the messages received.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received and
        >>> # the original feature for each node.
        >>> def node_udf(nodes):
        >>>     # nodes.data['h'] is a tensor of shape (N, 1),
        >>>     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
        >>>     # where N is the number of nodes in the batch, D is the number
        >>>     # of messages received per node for this node batch.
        >>>     return {'h': nodes.data['h'] + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[2.],
                [3.]])
        """
        return self._msgs

    def nodes(self):
        """Return the nodes in the batch.

        Returns
        -------
        NID : Tensor
            The IDs of the nodes in the batch. :math:`NID[i]` gives the ID of
            the i-th node.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received and
        >>> # the original ID for each node.
        >>> def node_udf(nodes):
        >>>     # nodes.nodes() is a tensor of shape (N),
        >>>     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
        >>>     # where N is the number of nodes in the batch, D is the number
        >>>     # of messages received per node for this node batch.
        >>>     return {'h': nodes.nodes().unsqueeze(-1).float()
        >>>         + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[1.],
                [3.]])
        """
        return self._nodes

    def batch_size(self):
        """Return the number of nodes in the batch.

        Returns
        -------
        int

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received for
        >>> # each node and increments the result by 1.
        >>> def node_udf(nodes):
        >>>     return {'h': torch.ones(nodes.batch_size(), 1)
        >>>         + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[2.],
                [3.]])
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
