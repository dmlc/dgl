"""Module for graph partition utilities."""
import time
import numpy as np

from ._ffi.function import _init_api
from .heterograph import DGLHeteroGraph
from . import backend as F
from . import utils
from .base import EID, NID

__all__ = ["metis_partition", "metis_partition_assignment",
           "partition_graph_with_halo"]


def reorder_nodes(g, new_node_ids):
    """ Generate a new graph with new node Ids.

    We assign each node in the input graph with a new node Id. This results in
    a new graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    new_node_ids : a tensor
        The new node Ids
    Returns
    -------
    DGLGraph
        The graph with new node Ids.
    """
    assert len(new_node_ids) == g.number_of_nodes(), \
        "The number of new node ids must match #nodes in the graph."
    new_node_ids = utils.toindex(new_node_ids)
    sorted_ids, idx = F.sort_1d(new_node_ids.tousertensor())
    assert F.asnumpy(sorted_ids[0]) == 0 \
        and F.asnumpy(sorted_ids[-1]) == g.number_of_nodes() - 1, \
        "The new node Ids are incorrect."
    new_gidx = _CAPI_DGLReorderGraph_Hetero(
        g._graph, new_node_ids.todgltensor())
    new_g = DGLHeteroGraph(gidx=new_gidx, ntypes=['_N'], etypes=['_E'])
    new_g.ndata['orig_id'] = idx
    return new_g


def _get_halo_heterosubgraph_inner_node(halo_subg):
    return _CAPI_GetHaloSubgraphInnerNodes_Hetero(halo_subg)


def partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle=False):
    '''Partition a graph.

    Based on the given node assignments for each partition, the function splits
    the input graph into subgraphs. A subgraph may contain HALO nodes which does
    not belong to the partition of a subgraph but are connected to the nodes
    in the partition within a fixed number of hops.

    If `reshuffle` is turned on, the function reshuffles node Ids and edge Ids
    of the input graph before partitioning. After reshuffling, all nodes and edges
    in a partition fall in a contiguous Id range in the input graph.
    The partitioend subgraphs have node data 'orig_id', which stores the node Ids
    in the original input graph.

    Parameters
    ------------
    g: DGLGraph
        The graph to be partitioned
    node_part: 1D tensor
        Specify which partition a node is assigned to. The length of this tensor
        needs to be the same as the number of nodes of the graph. Each element
        indicates the partition Id of a node.
    extra_cached_hops: int
        The number of hops a HALO node can be accessed.
    reshuffle : bool
        Resuffle nodes so that nodes in the same partition are in the same Id range.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition Id and the value is the DGLGraph of the partition.
    '''
    assert len(node_part) == g.number_of_nodes()
    node_part = utils.toindex(node_part)
    if reshuffle:
        start = time.time()
        node_part = node_part.tousertensor()
        sorted_part, new2old_map = F.sort_1d(node_part)
        new_node_ids = np.zeros((g.number_of_nodes(),), dtype=np.int64)
        new_node_ids[F.asnumpy(new2old_map)] = np.arange(
            0, g.number_of_nodes())
        g = reorder_nodes(g, new_node_ids)
        node_part = utils.toindex(sorted_part)
        # We reassign edges in in-CSR. In this way, after partitioning, we can ensure
        # that all edges in a partition are in the contiguous Id space.
        orig_eids = _CAPI_DGLReassignEdges_Hetero(g._graph, True)
        orig_eids = utils.toindex(orig_eids)
        orig_eids = orig_eids.tousertensor()
        orig_nids = g.ndata['orig_id']
        print('Reshuffle nodes and edges: {:.3f} seconds'.format(
            time.time() - start))

    start = time.time()
    subgs = _CAPI_DGLPartitionWithHalo_Hetero(
        g._graph, node_part.todgltensor(), extra_cached_hops)
    # g is no longer needed. Free memory.
    g = None
    print('Split the graph: {:.3f} seconds'.format(time.time() - start))
    subg_dict = {}
    node_part = node_part.tousertensor()
    start = time.time()

    # This creaets a subgraph from subgraphs returned from the CAPI above.
    def create_subgraph(subg, induced_nodes, induced_edges):
        subg1 = DGLHeteroGraph(gidx=subg.graph, ntypes=['_N'], etypes=['_E'])
        subg1.ndata[NID] = induced_nodes[0]
        subg1.edata[EID] = induced_edges[0]
        return subg1

    for i, subg in enumerate(subgs):
        inner_node = _get_halo_heterosubgraph_inner_node(subg)
        subg = create_subgraph(subg, subg.induced_nodes, subg.induced_edges)
        inner_node = F.zerocopy_from_dlpack(inner_node.to_dlpack())
        subg.ndata['inner_node'] = inner_node
        subg.ndata['part_id'] = F.gather_row(node_part, subg.ndata[NID])
        if reshuffle:
            subg.ndata['orig_id'] = F.gather_row(orig_nids, subg.ndata[NID])
            subg.edata['orig_id'] = F.gather_row(orig_eids, subg.edata[EID])

        if extra_cached_hops >= 1:
            inner_edge = F.zeros((subg.number_of_edges(),), F.int8, F.cpu())
            inner_nids = F.nonzero_1d(subg.ndata['inner_node'])
            # TODO(zhengda) we need to fix utils.toindex() to avoid the dtype cast below.
            inner_nids = F.astype(inner_nids, F.int64)
            inner_eids = subg.in_edges(inner_nids, form='eid')
            inner_edge = F.scatter_row(inner_edge, inner_eids,
                                       F.ones((len(inner_eids),), F.dtype(inner_edge), F.cpu()))
        else:
            inner_edge = F.ones((subg.number_of_edges(),), F.int8, F.cpu())
        subg.edata['inner_edge'] = inner_edge
        subg_dict[i] = subg
    print('Construct subgraphs: {:.3f} seconds'.format(time.time() - start))
    return subg_dict


def metis_partition_assignment(g, k, balance_ntypes=None, balance_edges=False):
    ''' This assigns nodes to different partitions with Metis partitioning algorithm.

    When performing Metis partitioning, we can put some constraint on the partitioning.
    Current, it supports two constrants to balance the partitioning. By default, Metis
    always tries to balance the number of nodes in each partition.

    * `balance_ntypes` balances the number of nodes of different types in each partition.
    * `balance_edges` balances the number of edges in each partition.

    To balance the node types, a user needs to pass a vector of N elements to indicate
    the type of each node. N is the number of nodes in the input graph.

    After the partition assignment, we construct partitions.

    Parameters
    ----------
    g : DGLGraph
        The graph to be partitioned
    k : int
        The number of partitions.
    balance_ntypes : tensor
        Node type of each node
    balance_edges : bool
        Indicate whether to balance the edges.

    Returns
    -------
    a 1-D tensor
        A vector with each element that indicates the partition Id of a vertex.
    '''
    # METIS works only on symmetric graphs.
    # The METIS runs on the symmetric graph to generate the node assignment to partitions.
    start = time.time()
    sym_gidx = _CAPI_DGLMakeSymmetric_Hetero(g._graph)
    sym_g = DGLHeteroGraph(gidx=sym_gidx)
    print('Convert a graph into a bidirected graph: {:.3f} seconds'.format(
        time.time() - start))
    vwgt = []
    # To balance the node types in each partition, we can take advantage of the vertex weights
    # in Metis. When vertex weights are provided, Metis will tries to generate partitions with
    # balanced vertex weights. A vertex can be assigned with multiple weights. The vertex weights
    # are stored in a vector of N * w elements, where N is the number of vertices and w
    # is the number of weights per vertex. Metis tries to balance the first weight, and then
    # the second weight, and so on.
    # When balancing node types, we use the first weight to indicate the first node type.
    # if a node belongs to the first node type, its weight is set to 1; otherwise, 0.
    # Similary, we set the second weight for the second node type and so on. The number
    # of weights is the same as the number of node types.
    start = time.time()
    if balance_ntypes is not None:
        assert len(balance_ntypes) == g.number_of_nodes(), \
            "The length of balance_ntypes should be equal to #nodes in the graph"
        balance_ntypes = F.tensor(balance_ntypes)
        uniq_ntypes = F.unique(balance_ntypes)
        for ntype in uniq_ntypes:
            vwgt.append(F.astype(balance_ntypes == ntype, F.int64))

    # When balancing edges in partitions, we use in-degree as one of the weights.
    if balance_edges:
        if balance_ntypes is None:
            vwgt.append(F.astype(g.in_degrees(), F.int64))
        else:
            for ntype in uniq_ntypes:
                nids = F.asnumpy(F.nonzero_1d(balance_ntypes == ntype))
                degs = np.zeros((g.number_of_nodes(),), np.int64)
                degs[nids] = F.asnumpy(g.in_degrees(nids))
                vwgt.append(F.zerocopy_from_numpy(degs))

    # The vertex weights have to be stored in a vector.
    if len(vwgt) > 0:
        vwgt = F.stack(vwgt, 1)
        shape = (np.prod(F.shape(vwgt),),)
        vwgt = F.reshape(vwgt, shape)
        vwgt = F.to_dgl_nd(vwgt)
        print(
            'Construct multi-constraint weights: {:.3f} seconds'.format(time.time() - start))
    else:
        vwgt = F.zeros((0,), F.int64, F.cpu())
        vwgt = F.to_dgl_nd(vwgt)

    start = time.time()
    node_part = _CAPI_DGLMetisPartition_Hetero(sym_g._graph, k, vwgt)
    print('Metis partitioning: {:.3f} seconds'.format(time.time() - start))
    if len(node_part) == 0:
        return None
    else:
        node_part = utils.toindex(node_part)
        return node_part.tousertensor()


def metis_partition(g, k, extra_cached_hops=0, reshuffle=False,
                    balance_ntypes=None, balance_edges=False):
    ''' This is to partition a graph with Metis partitioning.

    Metis assigns vertices to partitions. This API constructs subgraphs with the vertices assigned
    to the partitions and their incoming edges. A subgraph may contain HALO nodes which does
    not belong to the partition of a subgraph but are connected to the nodes
    in the partition within a fixed number of hops.

    When performing Metis partitioning, we can put some constraint on the partitioning.
    Current, it supports two constrants to balance the partitioning. By default, Metis
    always tries to balance the number of nodes in each partition.

    * `balance_ntypes` balances the number of nodes of different types in each partition.
    * `balance_edges` balances the number of edges in each partition.

    To balance the node types, a user needs to pass a vector of N elements to indicate
    the type of each node. N is the number of nodes in the input graph.

    If `reshuffle` is turned on, the function reshuffles node Ids and edge Ids
    of the input graph before partitioning. After reshuffling, all nodes and edges
    in a partition fall in a contiguous Id range in the input graph.
    The partitioend subgraphs have node data 'orig_id', which stores the node Ids
    in the original input graph.

    The partitioned subgraph is stored in DGLGraph. The DGLGraph has the `part_id`
    node data that indicates the partition a node belongs to. The subgraphs do not contain
    the node/edge data in the input graph.

    Parameters
    ------------
    g: DGLGraph
        The graph to be partitioned
    k: int
        The number of partitions.
    extra_cached_hops: int
        The number of hops a HALO node can be accessed.
    reshuffle : bool
        Resuffle nodes so that nodes in the same partition are in the same Id range.
    balance_ntypes : tensor
        Node type of each node
    balance_edges : bool
        Indicate whether to balance the edges.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition Id and the value is the DGLGraph of the partition.
    '''
    node_part = metis_partition_assignment(g, k, balance_ntypes, balance_edges)
    if node_part is None:
        return None

    # Then we split the original graph into parts based on the METIS partitioning results.
    return partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle)


_init_api("dgl.partition")
