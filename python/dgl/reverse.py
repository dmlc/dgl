from .graph import DGLGraph
from .batched_graph import BatchedDGLGraph

class ReversedGraph(DGLGraph):
    """See :func:`reverse` for the description.
    """
    def __init__(self, g):
        g_edge_list = g.edges()
        rg_edge_list = list(zip(g_edge_list[1], g_edge_list[0]))

        super(ReversedGraph, self).__init__(graph_data=rg_edge_list,
                                            node_frame=g._node_frame,
                                            edge_frame=g._edge_frame,
                                            multigraph=g.is_multigraph,
                                            readonly=g._readonly)

        self.register_apply_edge_func(g._apply_edge_func)
        self.register_apply_node_func(g._apply_node_func)
        self.register_message_func(g._message_func)
        self.register_reduce_func(g._reduce_func)

        self.set_reverse(g)

def reverse(g):
    """Return the reverse of a graph

    The reverse (also called converse, transpose) of a directed graph is
    another directed graph on the same nodes with edges reversed in terms of
    direction.

    Given a :class:`DGLGraph` object, we return another :class:`DGLGraph` object
    representing its reverse. This method is useful when dealing with undirected graphs.

    If the original graph has node features, its reverse will have the same features.
    If the original graph has edge features, a reversed edge will have the same feature
    as the original one. The features of the original graph and the reversed graph share
    memory.

    If the original graph has registered `apply_edge_func`, `apply_node_func`,
    `message_func` or `reduce_func`, the reverse will have the same.

    Note this function does not support :class:`~dgl.BatchedDGLGraph` objects.

    Examples
    --------
    Create a graph to reverse.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 1, 2], [1, 2, 0])
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.]])

    Reverse the graph and examine its structure.

    >>> rg = dgl.reverse(g)
    >>> print(rg)
    DGLGraph with 3 nodes and 3 edges.
    Node data: {'h': Scheme(shape=(1,), dtype=torch.float32)}
    Edge data: {'h': Scheme(shape=(1,), dtype=torch.float32)}

    The edges are reversed now.

    >>> rg.has_edges_between([1, 2, 0], [0, 1, 2])
    tensor([1, 1, 1])

    Reversed edges have the same feature as the original.

    >>> g.edges[[0, 2], [1, 0]].data['h'] == rg.edges[[1, 0], [0, 2]].data['h']
    tensor([[1],
            [1]], dtype=torch.uint8)

    The node/edge features of the reversed graph share memory with the original
    graph, which is helpful for both forward computation and back propagation.

    >>> g.ndata['h'] = g.ndata['h'] + 1
    >>> rg.ndata['h']
    tensor([[1.],
            [2.],
            [3.]])

    Once created, modify the topology and features of one graph will also affect
    those of its reverse.

    >>> rg.add_nodes(3, {'h': th.tensor([[6.], [7.], [8.]])})
    >>> g.number_of_nodes()
    6
    >>> g.ndata['h']
    tensor([[1.],
            [2.],
            [3.],
            [6.],
            [7.],
            [8.]])

    Parameters
    ----------
    g : dgl.DGLGraph
    """
    assert not isinstance(g, BatchedDGLGraph), 'reverse is not supported for a BatchedDGLGraph object'

    if g._reverse:
        return g._reverse
    rg = ReversedGraph(g)
    g.set_reverse(rg)
    return rg