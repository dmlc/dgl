#   Copyright (c) 2023, DGL Team
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

"""To block method."""

from collections import defaultdict
from collections.abc import Mapping

from .. import backend as F, utils
from ..base import DGLError
from ..heterograph import DGLBlock
from .._ffi.capi import *

__all__ = ["to_block"]


def to_block(g, dst_nodes=None, include_dst_in_src=True, src_nodes=None):
    """Convert a graph into a bipartite-structured *block* for message passing.

    A block is a graph consisting of two sets of nodes: the
    *source* nodes and *destination* nodes.  The source and destination nodes can have multiple
    node types.  All the edges connect from source nodes to destination nodes.

    Specifically, the source nodes and destination nodes will have the same node types as the
    ones in the original graph.  DGL maps each edge ``(u, v)`` with edge type
    ``(utype, etype, vtype)`` in the original graph to the edge with type
    ``etype`` connecting from node ID ``u`` of type ``utype`` in the source side to node
    ID ``v`` of type ``vtype`` in the destination side.

    For blocks returned by :func:`to_block`, the destination nodes of the block will only
    contain the nodes that have at least one inbound edge of any type.  The source nodes
    of the block will only contain the nodes that appear in the destination nodes, as well
    as the nodes that have at least one outbound edge connecting to one of the destination nodes.

    The destination nodes are specified by the :attr:`dst_nodes` argument if it is not None.

    Parameters
    ----------
    graph : DGLGraph
        The graph.  Can be either on CPU or GPU.
    dst_nodes : Tensor or dict[str, Tensor], optional
        The list of destination nodes.

        If a tensor is given, the graph must have only one node type.

        If given, it must be a superset of all the nodes that have at least one inbound
        edge.  An error will be raised otherwise.
    include_dst_in_src : bool
        If False, do not include destination nodes in source nodes.

        (Default: True)

    src_nodes : Tensor or disct[str, Tensor], optional
        The list of source nodes (and prefixed by destination nodes if
        `include_dst_in_src` is True).

        If a tensor is given, the graph must have only one node type.

    Returns
    -------
    DGLBlock
        The new graph describing the block.

        The node IDs induced for each type in both sides would be stored in feature
        ``dgl.NID``.

        The edge IDs induced for each type would be stored in feature ``dgl.EID``.

    Raises
    ------
    DGLError
        If :attr:`dst_nodes` is specified but it is not a superset of all the nodes that
        have at least one inbound edge.

        If :attr:`dst_nodes` is not None, and :attr:`g` and :attr:`dst_nodes`
        are not in the same context.

    Notes
    -----
    :func:`to_block` is most commonly used in customizing neighborhood sampling
    for stochastic training on a large graph.  Please refer to the user guide
    :ref:`guide-minibatch` for a more thorough discussion about the methodology
    of stochastic training.

    See also :func:`create_block` for more flexible construction of blocks.

    Examples
    --------
    Converting a homogeneous graph to a block as described above:

    >>> g = dgl.graph(([1, 2], [2, 3]))
    >>> block = dgl.to_block(g, torch.LongTensor([3, 2]))

    The destination nodes would be exactly the same as the ones given: [3, 2].

    >>> induced_dst = block.dstdata[dgl.NID]
    >>> induced_dst
    tensor([3, 2])

    The first few source nodes would also be exactly the same as
    the ones given.  The rest of the nodes are the ones necessary for message passing
    into nodes 3, 2.  This means that the node 1 would be included.

    >>> induced_src = block.srcdata[dgl.NID]
    >>> induced_src
    tensor([3, 2, 1])

    You can notice that the first two nodes are identical to the given nodes as well as
    the destination nodes.

    The induced edges can also be obtained by the following:

    >>> block.edata[dgl.EID]
    tensor([2, 1])

    This indicates that edge (2, 3) and (1, 2) are included in the result graph.  You can
    verify that the first edge in the block indeed maps to the edge (2, 3), and the
    second edge in the block indeed maps to the edge (1, 2):

    >>> src, dst = block.edges(order='eid')
    >>> induced_src[src], induced_dst[dst]
    (tensor([2, 1]), tensor([3, 2]))

    The destination nodes specified must be a superset of the nodes that have edges connecting
    to them.  For example, the following will raise an error since the destination nodes
    does not contain node 3, which has an edge connecting to it.

    >>> g = dgl.graph(([1, 2], [2, 3]))
    >>> dgl.to_block(g, torch.LongTensor([2]))     # error

    Converting a heterogeneous graph to a block is similar, except that when specifying
    the destination nodes, you have to give a dict:

    >>> g = dgl.heterograph({('A', '_E', 'B'): ([1, 2], [2, 3])})

    If you don't specify any node of type A on the destination side, the node type ``A``
    in the block would have zero nodes on the destination side.

    >>> block = dgl.to_block(g, {'B': torch.LongTensor([3, 2])})
    >>> block.number_of_dst_nodes('A')
    0
    >>> block.number_of_dst_nodes('B')
    2
    >>> block.dstnodes['B'].data[dgl.NID]
    tensor([3, 2])

    The source side would contain all the nodes on the destination side:

    >>> block.srcnodes['B'].data[dgl.NID]
    tensor([3, 2])

    As well as all the nodes that have connections to the nodes on the destination side:

    >>> block.srcnodes['A'].data[dgl.NID]
    tensor([2, 1])

    See also
    --------
    create_block
    """
    if dst_nodes is None:
        # Find all nodes that appeared as destinations
        dst_nodes = defaultdict(list)
        for etype in g.canonical_etypes:
            _, dst = g.edges(etype=etype)
            dst_nodes[etype[2]].append(dst)
        dst_nodes = {
            ntype: F.unique(F.cat(values, 0))
            for ntype, values in dst_nodes.items()
        }
    elif not isinstance(dst_nodes, Mapping):
        # dst_nodes is a Tensor, check if the g has only one type.
        if len(g.ntypes) > 1:
            raise DGLError(
                "Graph has more than one node type; please specify a dict for dst_nodes."
            )
        dst_nodes = {g.ntypes[0]: dst_nodes}

    dst_node_ids = [
        utils.toindex(dst_nodes.get(ntype, []), g._idtype_str).tousertensor(
            ctx=F.to_backend_ctx(g._graph.ctx)
        )
        for ntype in g.ntypes
    ]
    dst_node_ids_nd = [F.to_dgl_nd(nodes) for nodes in dst_node_ids]

    for d in dst_node_ids_nd:
        if g._graph.ctx != d.ctx:
            raise ValueError("g and dst_nodes need to have the same context.")

    src_node_ids = None
    src_node_ids_nd = None
    if src_nodes is not None and not isinstance(src_nodes, Mapping):
        # src_nodes is a Tensor, check if the g has only one type.
        if len(g.ntypes) > 1:
            raise DGLError(
                "Graph has more than one node type; please specify a dict for src_nodes."
            )
        src_nodes = {g.ntypes[0]: src_nodes}
        src_node_ids = [
            F.copy_to(
                F.tensor(src_nodes.get(ntype, []), dtype=g.idtype),
                F.to_backend_ctx(g._graph.ctx),
            )
            for ntype in g.ntypes
        ]
        src_node_ids_nd = [F.to_dgl_nd(nodes) for nodes in src_node_ids]

        for d in src_node_ids_nd:
            if g._graph.ctx != d.ctx:
                raise ValueError(
                    "g and src_nodes need to have the same context."
                )
    else:
        # use an empty list to signal we need to generate it
        src_node_ids_nd = []

    new_graph_index, src_nodes_ids_nd, induced_edges_nd = _CAPI_DGLToBlock(
        g._graph, dst_node_ids_nd, include_dst_in_src, src_node_ids_nd
    )

    # The new graph duplicates the original node types to SRC and DST sets.
    new_ntypes = (g.ntypes, g.ntypes)
    new_graph = DGLBlock(new_graph_index, new_ntypes, g.etypes)
    assert new_graph.is_unibipartite  # sanity check

    src_node_ids = [F.from_dgl_nd(src) for src in src_nodes_ids_nd]
    edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges_nd]

    node_frames = utils.extract_node_subframes_for_block(
        g, src_node_ids, dst_node_ids
    )
    edge_frames = utils.extract_edge_subframes(g, edge_ids)
    utils.set_new_frames(
        new_graph, node_frames=node_frames, edge_frames=edge_frames
    )

    return new_graph
