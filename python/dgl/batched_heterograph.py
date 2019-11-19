"""Classes and functions for batching multiple heterographs together."""
from collections.abc import Iterable

from . import backend as F
from . import graph_index
from . import heterograph_index
from .base import ALL, is_all
from .frame import FrameRef, Frame
from .heterograph import DGLHeteroGraph

__all__ = ['BatchedDGLHeteroGraph', 'unbatch_hetero', 'batch_hetero']

class BatchedDGLHeteroGraph(DGLHeteroGraph):
    """Class for batched DGLHeteroGraphs.

    A :class:`BatchedDGLHeteroGraph` basically merges a list of small graphs into a giant
    graph so that one can perform message passing and readout over a batch of graphs
    simultaneously.

    For a given node/edge type, the nodes/edges are re-indexed with a new id in the
    batched graph with the rule below:

    ======  ==========  ========================  ===  ==========================
    item    Graph 1     Graph 2                   ...  Graph k
    ======  ==========  ========================  ===  ==========================
    raw id  0, ..., N1       0, ..., N2           ...  ..., Nk
    new id  0, ..., N1  N1 + 1, ..., N1 + N2 + 1  ...  ..., N1 + ... + Nk + k - 1
    ======  ==========  ========================  ===  ==========================

    To modify the features in :class:`BatchedDGLHeteroGraph` has no effect on the original
    graphs. See the examples below about how to work around.

    Parameters
    ----------
    graph_list : iterable
        A collection of :class:`~dgl.DGLHeteroGraph` to be batched.
    node_attrs : None or dict
        The node attributes to be batched. If ``None``, the resulted graph will not have
        features. If ``dict``, it maps str to str or iterable. The keys represent names of
        node types and the values represent the node features to be batched for the
        corresponding type. By default, we use all features for all types of nodes.
    edge_attrs : None or dict
        Same as for the case of :attr:`node_attrs`.

    Examples
    --------
    Create two :class:`~dgl.DGLHeteroGraph` objects.

    **Instantiation:**

    >>> import dgl
    >>> import torch as th
    >>> g1 = dgl.heterograph({
    ...    ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    ...    ('user', 'plays', 'game'): [(0, 0), (1, 0)]
    ... })
    >>> g1.nodes['user'].data['h1'] = th.tensor([[0.], [1.], [2.]])
    >>> g1.nodes['user'].data['h2'] = th.tensor([[3.], [4.], [5.]])
    >>> g1.nodes['game'].data['h1'] = th.tensor([[0.]])
    >>> g1.edges['plays'].data['h1'] = th.tensor([[0.], [1.]])

    >>> g2 = dgl.heterograph({
    ...    ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    ...    ('user', 'plays', 'game'): [(0, 0), (1, 0)]
    ... })
    >>> g2.nodes['user'].data['h1'] = th.tensor([[0.], [1.], [2.]])
    >>> g2.nodes['user'].data['h2'] = th.tensor([[3.], [4.], [5.]])
    >>> g2.nodes['game'].data['h1'] = th.tensor([[0.]])
    >>> g2.edges['plays'].data['h1'] = th.tensor([[0.], [1.]])

    Merge two :class:`~dgl.DGLHeteroGraph` objects into one :class:`BatchedDGLHeteroGraph` object.
    When merging a list of graphs, we can choose to include only a subset of the attributes.

    >>> # For edge types, only canonical edge types are allowed to avoid ambiguity.
    >>> bg = dgl.batch_hetero([g1, g2], node_attrs={'user': ['h1', 'h2'], 'game': None},
    ...                       edge_attrs={('user', 'plays', 'game'): 'h1'})

    Below one can see that the nodes are re-indexed. The edges are re-indexed in
    the same way.

    >>> bg.nodes('user')
    tensor([0, 1, 2, 3, 4, 5])
    >>> bg.nodes['user'].data['h1']
    tensor([[0.],
            [1.],
            [2.],
            [0.],
            [1.],
            [2.]])

    **Property:**

    We can still get a brief summary of the graphs that constitute the batched graph.

    >>> bg.batch_size
    2
    >>> bg.batch_num_nodes('user')
    [3, 3]
    >>> bg.batch_num_edges(('user', 'plays', 'game'))
    [2, 2]

    **Update Attributes:**

    Updating the attributes of the batched graph has no effect on the original graphs.

    >>> bg.nodes['game'].data['h1'] = th.tensor([[1.], [1.]])
    >>> g2.nodes['game'].data['h1']
    tensor([[0.]])

    Instead, we can decompose the batched graph back into a list of graphs and use them
    to replace the original graphs.

    >>> g3, g4 = dgl.unbatch_hetero(bg) # returns a list of DGLHeteroGraph objects
    >>> g4.nodes['game'].data['h1']
    tensor([[1.]])
    """
    def __init__(self, graph_list, node_attrs, edge_attrs):
        # Sanity check. Make sure all graphs have the same node/edge types, in the same order.
        ref_graph = graph_list[0]
        ref_canonical_etypes = ref_graph.canonical_etypes
        ref_ntypes = ref_graph.ntypes
        ref_etypes = ref_graph.etypes
        for i in range(1, len(graph_list)):
            g_i = graph_list[i]
            g_i_canonical_etypes = g_i.canonical_etypes
            g_i_ntypes = g_i.ntypes
            assert len(g_i_canonical_etypes) == len(ref_canonical_etypes), \
                'Got different numbers of canonical etypes for graphs in the same batch.'
            assert len(g_i_ntypes) == len(ref_ntypes), \
                'Got different number of node types for graphs in the same batch.'
            for tid, ety in enumerate(ref_canonical_etypes):
                assert ety == g_i_canonical_etypes[tid], \
                    'Got different canonical etype with id {:d} ' \
                    'for graphs in the same batch.'.format(tid)
            for tid, nty in enumerate(ref_ntypes):
                assert nty == g_i_ntypes[tid], \
                    'Got different node type with id {:d} ' \
                    'for graphs in the same batch.'.format(tid)

        # Merge node and edge frames
        def _get_num_item_and_attr_types(g, mode, mode_type):
            if mode == 'node':
                num_items = g.number_of_nodes(mode_type)
                attr_types = set(g.node_attr_schemes(mode_type).keys())
            elif mode == 'edge':
                num_items = g.number_of_edges(mode_type)
                attr_types = set(g.edge_attr_schemes(mode_type).keys())
            return num_items, attr_types

        def _init_attrs(types, attrs, mode):
            formatted_attrs = {t: set() for t in types}
            if is_all(attrs):
                ref_g_indices = dict()
                for typ in types:
                    # Check if at least a graph has mode items and
                    # associated features for a particular type.
                    for i, g in enumerate(graph_list):
                        g_num_items, g_attrs = _get_num_item_and_attr_types(g, mode, typ)
                        if g_num_items > 0 and len(g_attrs) > 0:
                            formatted_attrs[typ] = g_attrs
                            ref_g_indices[typ] = i
                            break
                for typ in types:
                    # Check if all the graphs with mode items of a particular type
                    # have the same associated features.
                    if len(formatted_attrs[typ]) > 0:
                        for i, g in enumerate(graph_list):
                            g_num_items, g_attrs = _get_num_item_and_attr_types(g, mode, typ)
                            if g_attrs != formatted_attrs[typ] and g_num_items > 0:
                                raise ValueError(
                                    'Expect graph {0} and {1} to have the same {2} attributes '
                                    'for type {3} when {2}_attrs=ALL, got {4} and {5}.'.format(
                                        ref_g_indices[typ], i, mode, typ, attrs, g_attrs))
            elif isinstance(attrs, dict):
                for typ, v in attrs.items():
                    if isinstance(v, str):
                        formatted_attrs[typ] = [v]
                    elif isinstance(v, Iterable):
                        formatted_attrs[typ] = list(v)
                    elif v is not None:
                        raise ValueError('Expected {} attrs for type {} to be str '
                                         'or iterable, got {}'.format(mode, typ, type(v)))
            elif attrs is not None:
                raise ValueError('Expected {} attrs to be of type None or dict,'
                                 'got type {}'.format(mode, type(attrs)))
            return formatted_attrs

        node_attrs = _init_attrs(ref_ntypes, node_attrs, 'node')
        edge_attrs = _init_attrs(ref_canonical_etypes, edge_attrs, 'edge')

        node_frames = []
        for tid, typ in enumerate(ref_ntypes):
            if len(node_attrs[typ]) == 0:
                # Emtpy frames will be created when we instantiate a DGLHeteroGraph.
                node_frames.append(None)
            else:
                # NOTE: following code will materialize the columns of the input graphs.
                cols = {key: F.cat([gr._node_frames[tid][key] for gr in graph_list
                                    if gr.number_of_nodes(typ) > 0], dim=0)
                        for key in node_attrs[typ]}
                node_frames.append(FrameRef(Frame(cols)))

        edge_frames = []
        for tid, typ in enumerate(ref_canonical_etypes):
            if len(edge_attrs[typ]) == 0:
                # Emtpy frames will be created when we instantiate a DGLHeteroGraph.
                edge_frames.append(None)
            else:
                # NOTE: following code will materialize the columns of the input graphs.
                cols = {key: F.cat([gr._edge_frames[tid][key] for gr in graph_list
                                    if gr.number_of_edges(typ) > 0], dim=0)
                        for key in edge_attrs[typ]}
                edge_frames.append(FrameRef(Frame(cols)))

        # Create graph index for the batched graph
        src, dst, _ = graph_list[0]._graph.metagraph.edges()
        meta_edges = list(zip(list(src.tonumpy()), list(dst.tonumpy())))
        metagraph = graph_index.from_edge_list(meta_edges, True, True)
        batched_index = heterograph_index.disjoint_union(
            metagraph, [g._graph for g in graph_list])
        super(BatchedDGLHeteroGraph, self).__init__(gidx=batched_index,
                                                    ntypes=ref_ntypes,
                                                    etypes=ref_etypes,
                                                    node_frames=node_frames,
                                                    edge_frames=edge_frames)

        # extra members
        self._batch_size = 0
        # Store number of nodes/edge based on the id of node/edge types as we need
        # to handle both edge type and canonical edge type.
        self._batch_num_nodes = [[] for _ in range(len(ref_ntypes))]
        self._batch_num_edges = [[] for _ in range(len(ref_etypes))]

        for grh in graph_list:
            if isinstance(grh, BatchedDGLHeteroGraph):
                # Handle input graphs that are already batched
                self._batch_size += grh._batch_size
                for ntype_id in range(len(ref_ntypes)):
                    self._batch_num_nodes[ntype_id].extend(grh._batch_num_nodes[ntype_id])
                for etype_id in range(len(ref_etypes)):
                    self._batch_num_edges[etype_id].extend(grh._batch_num_edges[etype_id])
            else:
                self._batch_size += 1
                for ntype_id in range(len(ref_ntypes)):
                    self._batch_num_nodes[ntype_id].append(grh._graph.number_of_nodes(ntype_id))
                for etype_id in range(len(ref_etypes)):
                    self._batch_num_edges[etype_id].append(grh._graph.number_of_edges(etype_id))

    @property
    def batch_size(self):
        """Number of graphs in this batch.

        Returns
        -------
        int
            Number of graphs in this batch."""
        return self._batch_size

    def batch_num_nodes(self, ntype=None):
        """Return the numbers of nodes of the given type for all heterographs in the batch.

        Parameters
        ----------
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph. (Default: None)

        Returns
        -------
        list of int
            The ith element gives the number of nodes of the specified type in the ith graph.

        Examples
        --------

        >>> g1 = dgl.heterograph({
        ...      ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ...      ('user', 'plays', 'game'): [(0, 0), (1, 0), (2, 1), (3, 1)]
        ...      })
        >>> g2 = dgl.heterograph({
        ...      ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ...      ('user', 'plays', 'game'): [(0, 0), (1, 0), (2, 1)]
        ...      })
        >>> bg = dgl.batch_hetero([g1, g2])
        >>> bg.batch_num_nodes('user')
        [4, 3]
        >>> bg.batch_num_nodes('game')
        [2, 2]
        """
        return self._batch_num_nodes[self.get_ntype_id(ntype)]

    def batch_num_edges(self, etype=None):
        """Return the numbers of edges of the given type for all heterographs in the batch.

        Parameters
        ----------
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

        Returns
        -------
        list of int
            The ith element gives the number of edges of the specified type in the ith graph.

        Examples
        --------

        >>> g1 = dgl.heterograph({
        ...      ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ...      ('user', 'follows', 'developer'): [(0, 1), (1, 2)],
        ...      ('user', 'plays', 'game'): [(0, 0), (1, 0), (2, 1), (3, 1)]
        ...      })
        >>> g2 = dgl.heterograph({
        ...      ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ...      ('user', 'follows', 'developer'): [(0, 1), (1, 2)],
        ...      ('user', 'plays', 'game'): [(0, 0), (1, 0), (2, 1)]
        ...      })
        >>> bg = dgl.batch_hetero([g1, g2])
        >>> bg.batch_num_edges('plays')
        [4, 3]
        >>> # 'follows' is ambiguous and we use ('user', 'follows', 'user') instead.
        >>> bg.batch_num_edges(('user', 'follows', 'user'))
        [2, 2]
        """
        return self._batch_num_edges[self.get_etype_id(etype)]

def unbatch_hetero(graph):
    """Return the list of heterographs in this batch.

    Parameters
    ----------
    graph : BatchedDGLHeteroGraph
        The batched heterograph.

    Returns
    -------
    list
        A list of :class:`~dgl.BatchedDGLHeteroGraph` objects whose attributes are
        obtained by partitioning the attributes of the :attr:`graph`. The length of
        the list is the same as the batch size of :attr:`graph`.

    Notes
    -----
    Unbatching will break each field tensor of the batched graph into smaller
    partitions.

    For simpler tasks such as node/edge state aggregation, try to slice graphs along
    edge types and use readout functions.

    See Also
    --------
    batch_hetero
    """
    assert isinstance(graph, BatchedDGLHeteroGraph), \
        'Expect the input to be of type BatchedDGLHeteroGraph, got type {}'.format(type(graph))
    bsize = graph.batch_size
    bnn_all_types = graph._batch_num_nodes
    bne_all_types = graph._batch_num_edges
    ntypes = graph._ntypes
    etypes = graph._etypes
    node_frames = [[FrameRef(Frame(num_rows=bnn_all_types[tid][i])) for tid in range(len(ntypes))]
                   for i in range(bsize)]
    edge_frames = [[FrameRef(Frame(num_rows=bne_all_types[tid][i])) for tid in range(len(etypes))]
                   for i in range(bsize)]
    for tid in range(len(ntypes)):
        for attr, col in graph._node_frames[tid].items():
            col_splits = F.split(col, bnn_all_types[tid], dim=0)
            for i in range(bsize):
                node_frames[i][tid][attr] = col_splits[i]
    for tid in range(len(etypes)):
        for attr, col in graph._edge_frames[tid].items():
            col_splits = F.split(col, bne_all_types[tid], dim=0)
            for i in range(bsize):
                edge_frames[i][tid][attr] = col_splits[i]
    unbatched_graph_indices = heterograph_index.disjoint_partition(
        graph._graph, bnn_all_types, bne_all_types)
    return [DGLHeteroGraph(gidx=unbatched_graph_indices[i],
                           ntypes=ntypes,
                           etypes=etypes,
                           node_frames=node_frames[i],
                           edge_frames=edge_frames[i]) for i in range(bsize)]

def batch_hetero(graph_list, node_attrs=ALL, edge_attrs=ALL):
    """Batch a collection of :class:`~dgl.DGLHeteroGraph` and return a
    :class:`BatchedDGLHeteroGraph` object that is independent of the :attr:`graph_list`.

    Parameters
    ----------
    graph_list : iterable
        A collection of :class:`~dgl.DGLHeteroGraph` to be batched.
    node_attrs : None or dict
        The node attributes to be batched. If ``None``, the resulted graph will not have
        features. If ``dict``, it maps str to str or iterable. The keys represent names of
        node types and the values represent the node features to be batched for the
        corresponding type. By default, we use all features for all types of nodes.
    edge_attrs : None or dict
        Same as for the case of :attr:`node_attrs`.

    Returns
    -------
    BatchedDGLHeteroGraph
        One single batched heterograph

    See Also
    --------
    BatchedDGLHeteroGraph
    unbatch_hetero
    """
    return BatchedDGLHeteroGraph(graph_list, node_attrs, edge_attrs)
