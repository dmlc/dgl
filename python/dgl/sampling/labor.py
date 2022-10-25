#
#   Copyright (c) 2022 by Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Based off of neighbor.py
#


"""Labor sampling APIs"""

from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, NID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils
from .utils import EidExcluder
from ..backend import from_dgl_nd

__all__ = ["sample_labors"]


def sample_labors(
    g,
    nodes,
    fanout,
    random_seed,
    edge_dir="in",
    prob=None,
    importance_sampling=0,
    copy_ndata=True,
    copy_edata=True,
    exclude_edges=None,
    output_device=None,
):
    """Sample neighboring edges of the given nodes and return the induced subgraph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The graph returned will then contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLGraph
        The graph, allowed to have multiple node or edge types. Can be either on CPU or GPU.
    nodes : tensor or dict
        Node IDs to sample neighbors from.

        This argument can take a single ID tensor or a dictionary of node types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    fanout : int or dict[etype, int]
        The number of edges to be sampled for each node on each edge type.

        This argument can take a single int or a dictionary of edge types and ints.
        If a single int is given, DGL will sample this number of edges for each node for
        every edge type.

        If -1 is given for a single edge type, all the neighboring edges with that edge
        type will be selected.
    random_seed : tensor with 3 or 4 integers
        The random seed(s) to be used for sampling along with batch dependency count information
        stores as the last two elements
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    prob : str, optional
        Feature name used as the (unnormalized) probabilities associated with each
        neighboring edge of a node.  The feature must have only one element for each
        edge.

        The features must be non-negative floats, and the sum of the features of
        inbound/outbound edges for every node must be positive (though they don't have
        to sum up to one).  Otherwise, the result will be undefined.

        If :attr:`prob` is not None, GPU sampling is not supported.
    importance_sampling : bool, optional
        Whether to use importance sampling with replacement or uniform sampling without replacement.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the new graph are copied from
        the original graph.  If False, the new graph will not have any
        edge features.

        (Default: True)
    exclude_edges: tensor or dict
        Edge IDs to exclude during sampling neighbors for the seed nodes.

        This argument can take a single ID tensor or a dictionary of edge types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True, same tensors are used as
    the node or edge features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.

    Examples
    --------
    Assume that you have the following graph

    >>> g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))

    And the weights

    >>> g.edata['prob'] = torch.FloatTensor([0., 1., 0., 1., 0., 1.])

    To sample one inbound edge for node 0 and node 1:

    >>> sg = dgl.sampling.sample_labors(g, [0, 1], 1)
    >>> sg.edges(order='eid')
    (tensor([1, 0]), tensor([0, 1]))
    >>> sg.edata[dgl.EID]
    tensor([2, 0])

    To sample one inbound edge for node 0 and node 1 with probability in edge feature
    ``prob``:

    >>> sg = dgl.sampling.sample_labors(g, [0, 1], 1, prob='prob')
    >>> sg.edges(order='eid')
    (tensor([2, 1]), tensor([0, 1]))

    With ``fanout`` greater than the number of actual neighbors and without replacement,
    DGL will take all neighbors instead:

    >>> sg = dgl.sampling.sample_labors(g, [0, 1], 3)
    >>> sg.edges(order='eid')
    (tensor([1, 2, 0, 1]), tensor([0, 0, 1, 1]))

    To exclude certain EID's during sampling for the seed nodes:

    >>> g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))
    >>> g_edges = g.all_edges(form='all')``
    (tensor([0, 0, 1, 1, 2, 2]), tensor([1, 2, 0, 1, 2, 0]), tensor([0, 1, 2, 3, 4, 5]))
    >>> sg = dgl.sampling.sample_labors(g, [0, 1], 3, exclude_edges=[0, 1, 2])
    >>> sg.all_edges(form='all')
    (tensor([2, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> sg.has_edges_between(g_edges[0][:3],g_edges[1][:3])
    tensor([False, False, False])
    >>> g = dgl.heterograph({
    ...   ('drug', 'interacts', 'drug'): ([0, 0, 1, 1, 3, 2], [1, 2, 0, 1, 2, 0]),
    ...   ('drug', 'interacts', 'gene'): ([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]),
    ...   ('drug', 'treats', 'disease'): ([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0])})
    >>> g_edges = g.all_edges(form='all', etype=('drug', 'interacts', 'drug'))
    (tensor([0, 0, 1, 1, 3, 2]), tensor([1, 2, 0, 1, 2, 0]), tensor([0, 1, 2, 3, 4, 5]))
    >>> excluded_edges  = {('drug', 'interacts', 'drug'): g_edges[2][:3]}
    >>> sg = dgl.sampling.sample_labors(g, {'drug':[0, 1]}, 3, exclude_edges=excluded_edges)
    >>> sg.all_edges(form='all', etype=('drug', 'interacts', 'drug'))
    (tensor([2, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> sg.has_edges_between(g_edges[0][:3],g_edges[1][:3],etype=('drug', 'interacts', 'drug'))
    tensor([False, False, False])

    """
    if F.device_type(g.device) == "cpu" and not g.is_pinned():
        frontier, importances = _sample_labors(
            g,
            nodes,
            fanout,
            random_seed,
            edge_dir=edge_dir,
            prob=prob,
            importance_sampling=importance_sampling,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
            exclude_edges=exclude_edges,
        )
    else:
        frontier, importances = _sample_labors(
            g,
            nodes,
            fanout,
            random_seed,
            edge_dir=edge_dir,
            prob=prob,
            importance_sampling=importance_sampling,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
        )
        if exclude_edges is not None:
            eid_excluder = EidExcluder(exclude_edges)
            if importance_sampling != 0:
                frontier, importances = eid_excluder(frontier, importances)
            else:
                frontier = eid_excluder(frontier)
    return (
        (frontier, importances)
        if output_device is None
        else (
            frontier.to(output_device),
            list(map(lambda x: x.to(output_device), importances)),
        )
    )


def _sample_labors(
    g,
    nodes,
    fanout,
    random_seed,
    edge_dir="in",
    prob=None,
    importance_sampling=0,
    copy_ndata=True,
    copy_edata=True,
    exclude_edges=None,
):
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        nodes = {g.ntypes[0]: nodes}

    nodes = utils.prepare_tensor_dict(g, nodes, "nodes")
    if len(nodes) == 0:
        raise ValueError(
            "Got an empty dictionary in the nodes argument. "
            "Please pass in a dictionary with empty tensors as values instead."
        )
    ctx = utils.to_dgl_context(F.context(next(iter(nodes.values()))))
    nodes_all_types = []
    nids_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
            try:
                if len(g.ntypes) > 1:
                    nids_all_types.append(F.to_dgl_nd(g.ndata[NID][ntype]))
                else:
                    nids_all_types.append(F.to_dgl_nd(g.ndata[NID]))
            except KeyError:
                nids_all_types.append(nd.array([], ctx=ctx))
        else:
            nodes_all_types.append(nd.array([], ctx=ctx))
            nids_all_types.append(nd.array([], ctx=ctx))

    if isinstance(fanout, nd.NDArray):
        fanout_array = fanout
    else:
        if not isinstance(fanout, dict):
            fanout_array = [int(fanout)] * len(g.etypes)
        else:
            if len(fanout) != len(g.etypes):
                raise DGLError(
                    "Fan-out must be specified for each edge type "
                    "if a dict is provided."
                )
            fanout_array = [None] * len(g.etypes)
            for etype, value in fanout.items():
                fanout_array[g.get_etype_id(etype)] = value
        fanout_array = F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64))

    if (
        isinstance(prob, list)
        and len(prob) > 0
        and isinstance(prob[0], nd.NDArray)
    ):
        prob_arrays = prob
    elif prob is None:
        prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        prob_arrays = []
        for etype in g.canonical_etypes:
            if prob in g.edges[etype].data:
                prob_arrays.append(F.to_dgl_nd(g.edges[etype].data[prob]))
            else:
                prob_arrays.append(nd.array([], ctx=nd.cpu()))

    excluded_edges_all_t = []
    if exclude_edges is not None:
        if not isinstance(exclude_edges, dict):
            if len(g.etypes) > 1:
                raise DGLError(
                    "Must specify etype type when the graph is not homogeneous."
                )
            exclude_edges = {g.canonical_etypes[0]: exclude_edges}
        exclude_edges = utils.prepare_tensor_dict(g, exclude_edges, "edges")
        for etype in g.canonical_etypes:
            if etype in exclude_edges:
                excluded_edges_all_t.append(F.to_dgl_nd(exclude_edges[etype]))
            else:
                excluded_edges_all_t.append(nd.array([], ctx=ctx))

    ret_val = _CAPI_DGLSampleLabors(
        g._graph,
        nids_all_types,
        nodes_all_types,
        fanout_array,
        random_seed[0],
        random_seed[1],
        edge_dir,
        prob_arrays,
        excluded_edges_all_t,
        importance_sampling,
    )
    subgidx = ret_val[0]
    importances = [from_dgl_nd(importance) for importance in ret_val[1:]]
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)

    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(ret, node_frames=node_frames)

    if copy_edata:
        edge_frames = utils.extract_edge_subframes(g, induced_edges)
        utils.set_new_frames(ret, edge_frames=edge_frames)

    return ret, importances


DGLHeteroGraph.sample_labors = utils.alias_func(sample_labors)

_init_api("dgl.sampling.labor", __name__)
