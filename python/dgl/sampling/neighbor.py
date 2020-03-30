"""Neighbor sampling APIs"""

from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils

__all__ = [
    'sample_neighbors',
    'select_topk']

def sample_neighbors(g, nodes, fanout, edge_dir='in', prob=None, replace=False):
    """Sample from the neighbors of the given nodes and return the induced subgraph.

    When sampling with replacement, the sampled subgraph could have parallel edges.

    For sampling without replace, if fanout > the number of neighbors, all the
    neighbors are sampled.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.
    fanout : int or dict[etype, int]
        The number of sampled neighbors for each node on each edge type. Provide a dict
        to specify different fanout values for each edge type.
    edge_dir : str, optional
        Edge direction ('in' or 'out'). If is 'in', sample from in edges. Otherwise,
        sample from out edges.
    prob : str, optional
        Feature name used as the probabilities associated with each neighbor of a node.
        Its shape should be compatible with a scalar edge feature tensor.
    replace : bool, optional
        If True, sample with replacement.

    Returns
    -------
    DGLHeteroGraph
        A sampled subgraph containing only the sampled neighbor edges from
        ``nodes``. The sampled subgraph has the same metagraph as the original
        one.
    """
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(utils.toindex(nodes[ntype]).todgltensor())
        else:
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(fanout, dict):
        fanout_array = [int(fanout)] * len(g.etypes)
    else:
        if len(fanout) != len(g.etypes):
            raise DGLError('Fan-out must be specified for each edge type '
                           'if a dict is provided.')
        fanout_array = [None] * len(g.etypes)
        for etype, value in fanout.items():
            fanout_array[g.get_etype_id(etype)] = value
    fanout_array = utils.toindex(fanout_array).todgltensor()

    if prob is None:
        prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        prob_arrays = []
        for etype in g.canonical_etypes:
            if prob in g.edges[etype].data:
                prob_arrays.append(F.zerocopy_to_dgl_ndarray(g.edges[etype].data[prob]))
            else:
                prob_arrays.append(nd.array([], ctx=nd.cpu()))

    subgidx = _CAPI_DGLSampleNeighbors(g._graph, nodes_all_types, fanout_array,
                                       edge_dir, prob_arrays, replace)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i].tousertensor()
    return ret

def select_topk(g, k, weight, nodes=None, edge_dir='in', ascending=False):
    """Select the neighbors with k-largest weights on the connecting edges for each given node.

    If k > the number of neighbors, all the neighbors are sampled.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    k : int or dict[etype, int]
        The K value.
    weight : str
        Feature name of the weights associated with each edge. Its shape should be
        compatible with a scalar edge feature tensor.
    nodes : tensor or dict, optional
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id
        tensor if the given graph g has only one type of nodes.
    edge_dir : str, optional
        Edge direction ('in' or 'out'). If is 'in', sample from in edges.
        Otherwise, sample from out edges.
    ascending : bool, optional
        If true, elements are sorted by ascending order, equivalent to find
        the K smallest values. Otherwise, find K largest values.

    Returns
    -------
    DGLHeteroGraph
        A sampled subgraph by top k criterion. The sampled subgraph has the same
        metagraph as the original one.
    """
    # Rectify nodes to a dictionary
    if nodes is None:
        nodes = {ntype: F.arange(0, g.number_of_nodes(ntype)) for ntype in g.ntypes}
    elif not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}

    # Parse nodes into a list of NDArrays.
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(utils.toindex(nodes[ntype]).todgltensor())
        else:
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(k, dict):
        k_array = [int(k)] * len(g.etypes)
    else:
        if len(k) != len(g.etypes):
            raise DGLError('K value must be specified for each edge type '
                           'if a dict is provided.')
        k_array = [None] * len(g.etypes)
        for etype, value in k.items():
            k_array[g.get_etype_id(etype)] = value
    k_array = utils.toindex(k_array).todgltensor()

    weight_arrays = []
    for etype in g.canonical_etypes:
        if weight in g.edges[etype].data:
            weight_arrays.append(F.zerocopy_to_dgl_ndarray(g.edges[etype].data[weight]))
        else:
            raise DGLError('Edge weights "{}" do not exist for relation graph "{}".'.format(
                weight, etype))

    subgidx = _CAPI_DGLSampleNeighborsTopk(
        g._graph, nodes_all_types, k_array, edge_dir, weight_arrays, bool(ascending))
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i].tousertensor()
    return ret

_init_api('dgl.sampling.neighbor', __name__)
