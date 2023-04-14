"""Module for graph partition utilities."""
import os
import re
import time

import numpy as np

from . import backend as F, utils
from ._ffi.function import _init_api
from .base import EID, ETYPE, NID, NTYPE
from .heterograph import DGLGraph
from .ndarray import NDArray
from .subgraph import edge_subgraph

__all__ = [
    "metis_partition",
    "metis_partition_assignment",
    "partition_graph_with_halo",
]


def reorder_nodes(g, new_node_ids):
    """Generate a new graph with new node IDs.

    We assign each node in the input graph with a new node ID. This results in
    a new graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    new_node_ids : a tensor
        The new node IDs
    Returns
    -------
    DGLGraph
        The graph with new node IDs.
    """
    assert (
        len(new_node_ids) == g.num_nodes()
    ), "The number of new node ids must match #nodes in the graph."
    new_node_ids = utils.toindex(new_node_ids)
    sorted_ids, idx = F.sort_1d(new_node_ids.tousertensor())
    assert (
        F.asnumpy(sorted_ids[0]) == 0
        and F.asnumpy(sorted_ids[-1]) == g.num_nodes() - 1
    ), "The new node IDs are incorrect."
    new_gidx = _CAPI_DGLReorderGraph_Hetero(
        g._graph, new_node_ids.todgltensor()
    )
    new_g = DGLGraph(gidx=new_gidx, ntypes=["_N"], etypes=["_E"])
    new_g.ndata["orig_id"] = idx
    return new_g


def _get_halo_heterosubgraph_inner_node(halo_subg):
    return _CAPI_GetHaloSubgraphInnerNodes_Hetero(halo_subg)


def reshuffle_graph(g, node_part=None):
    """Reshuffle node ids and edge IDs of a graph.

    This function reshuffles nodes and edges in a graph so that all nodes/edges of the same type
    have contiguous IDs. If a graph is partitioned and nodes are assigned to different partitions,
    all nodes/edges in a partition should
    get contiguous IDs; within a partition, all nodes/edges of the same type have contigous IDs.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    node_part : Tensor
        This is a vector whose length is the same as the number of nodes in the input graph.
        Each element indicates the partition ID the corresponding node is assigned to.

    Returns
    -------
    (DGLGraph, Tensor)
        The graph whose nodes and edges are reshuffled.
        The 1D tensor that indicates the partition IDs of the nodes in the reshuffled graph.
    """
    # In this case, we don't need to reshuffle node IDs and edge IDs.
    if node_part is None:
        g.ndata["orig_id"] = F.arange(0, g.num_nodes())
        g.edata["orig_id"] = F.arange(0, g.num_edges())
        return g, None

    start = time.time()
    if node_part is not None:
        node_part = utils.toindex(node_part)
        node_part = node_part.tousertensor()
    if NTYPE in g.ndata:
        is_hetero = len(F.unique(g.ndata[NTYPE])) > 1
    else:
        is_hetero = False
    if is_hetero:
        num_node_types = F.max(g.ndata[NTYPE], 0) + 1
        if node_part is not None:
            sorted_part, new2old_map = F.sort_1d(
                node_part * num_node_types + g.ndata[NTYPE]
            )
        else:
            sorted_part, new2old_map = F.sort_1d(g.ndata[NTYPE])
        sorted_part = F.floor_div(sorted_part, num_node_types)
    elif node_part is not None:
        sorted_part, new2old_map = F.sort_1d(node_part)
    else:
        g.ndata["orig_id"] = g.ndata[NID]
        g.edata["orig_id"] = g.edata[EID]
        return g, None

    new_node_ids = np.zeros((g.num_nodes(),), dtype=np.int64)
    new_node_ids[F.asnumpy(new2old_map)] = np.arange(0, g.num_nodes())
    # If the input graph is homogneous, we only need to create an empty array, so that
    # _CAPI_DGLReassignEdges_Hetero knows how to handle it.
    etype = (
        g.edata[ETYPE]
        if ETYPE in g.edata
        else F.zeros((0), F.dtype(sorted_part), F.cpu())
    )
    g = reorder_nodes(g, new_node_ids)
    node_part = utils.toindex(sorted_part)
    # We reassign edges in in-CSR. In this way, after partitioning, we can ensure
    # that all edges in a partition are in the contiguous ID space.
    etype_idx = utils.toindex(etype)
    orig_eids = _CAPI_DGLReassignEdges_Hetero(
        g._graph, etype_idx.todgltensor(), node_part.todgltensor(), True
    )
    orig_eids = utils.toindex(orig_eids)
    orig_eids = orig_eids.tousertensor()
    g.edata["orig_id"] = orig_eids

    print(
        "Reshuffle nodes and edges: {:.3f} seconds".format(time.time() - start)
    )
    return g, node_part.tousertensor()


def partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle=False):
    """Partition a graph.

    Based on the given node assignments for each partition, the function splits
    the input graph into subgraphs. A subgraph may contain HALO nodes which does
    not belong to the partition of a subgraph but are connected to the nodes
    in the partition within a fixed number of hops.

    If `reshuffle` is turned on, the function reshuffles node IDs and edge IDs
    of the input graph before partitioning. After reshuffling, all nodes and edges
    in a partition fall in a contiguous ID range in the input graph.
    The partitioend subgraphs have node data 'orig_id', which stores the node IDs
    in the original input graph.

    Parameters
    ------------
    g: DGLGraph
        The graph to be partitioned
    node_part: 1D tensor
        Specify which partition a node is assigned to. The length of this tensor
        needs to be the same as the number of nodes of the graph. Each element
        indicates the partition ID of a node.
    extra_cached_hops: int
        The number of hops a HALO node can be accessed.
    reshuffle : bool
        Resuffle nodes so that nodes in the same partition are in the same ID range.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition ID and the value is the DGLGraph of the partition.
    Tensor
        1D tensor that stores the mapping between the reshuffled node IDs and
        the original node IDs if 'reshuffle=True'. Otherwise, return None.
    Tensor
        1D tensor that stores the mapping between the reshuffled edge IDs and
        the original edge IDs if 'reshuffle=True'. Otherwise, return None.
    """
    assert len(node_part) == g.num_nodes()
    if reshuffle:
        g, node_part = reshuffle_graph(g, node_part)
        orig_nids = g.ndata["orig_id"]
        orig_eids = g.edata["orig_id"]

    node_part = utils.toindex(node_part)
    start = time.time()
    subgs = _CAPI_DGLPartitionWithHalo_Hetero(
        g._graph, node_part.todgltensor(), extra_cached_hops
    )
    # g is no longer needed. Free memory.
    g = None
    print("Split the graph: {:.3f} seconds".format(time.time() - start))
    subg_dict = {}
    node_part = node_part.tousertensor()
    start = time.time()

    # This function determines whether an edge belongs to a partition.
    # An edge is assigned to a partition based on its destination node. If its destination node
    # is assigned to a partition, we assign the edge to the partition as well.
    def get_inner_edge(subg, inner_node):
        inner_edge = F.zeros((subg.num_edges(),), F.int8, F.cpu())
        inner_nids = F.nonzero_1d(inner_node)
        # TODO(zhengda) we need to fix utils.toindex() to avoid the dtype cast below.
        inner_nids = F.astype(inner_nids, F.int64)
        inner_eids = subg.in_edges(inner_nids, form="eid")
        inner_edge = F.scatter_row(
            inner_edge,
            inner_eids,
            F.ones((len(inner_eids),), F.dtype(inner_edge), F.cpu()),
        )
        return inner_edge

    # This creaets a subgraph from subgraphs returned from the CAPI above.
    def create_subgraph(subg, induced_nodes, induced_edges, inner_node):
        subg1 = DGLGraph(gidx=subg.graph, ntypes=["_N"], etypes=["_E"])
        # If IDs are shuffled, we should shuffled edges. This will help us collect edge data
        # from the distributed graph after training.
        if reshuffle:
            # When we shuffle edges, we need to make sure that the inner edges are assigned with
            # contiguous edge IDs and their ID range starts with 0. In other words, we want to
            # place these edge IDs in the front of the edge list. To ensure that, we add the IDs
            # of outer edges with a large value, so we will get the sorted list as we want.
            max_eid = F.max(induced_edges[0], 0) + 1
            inner_edge = get_inner_edge(subg1, inner_node)
            eid = F.astype(induced_edges[0], F.int64) + max_eid * F.astype(
                inner_edge == 0, F.int64
            )

            _, index = F.sort_1d(eid)
            subg1 = edge_subgraph(subg1, index, relabel_nodes=False)
            subg1.ndata[NID] = induced_nodes[0]
            subg1.edata[EID] = F.gather_row(induced_edges[0], index)
        else:
            subg1.ndata[NID] = induced_nodes[0]
            subg1.edata[EID] = induced_edges[0]
        return subg1

    for i, subg in enumerate(subgs):
        inner_node = _get_halo_heterosubgraph_inner_node(subg)
        inner_node = F.zerocopy_from_dlpack(inner_node.to_dlpack())
        subg = create_subgraph(
            subg, subg.induced_nodes, subg.induced_edges, inner_node
        )
        subg.ndata["inner_node"] = inner_node
        subg.ndata["part_id"] = F.gather_row(node_part, subg.ndata[NID])
        if reshuffle:
            subg.ndata["orig_id"] = F.gather_row(orig_nids, subg.ndata[NID])
            subg.edata["orig_id"] = F.gather_row(orig_eids, subg.edata[EID])

        if extra_cached_hops >= 1:
            inner_edge = get_inner_edge(subg, inner_node)
        else:
            inner_edge = F.ones((subg.num_edges(),), F.int8, F.cpu())
        subg.edata["inner_edge"] = inner_edge
        subg_dict[i] = subg
    print("Construct subgraphs: {:.3f} seconds".format(time.time() - start))
    if reshuffle:
        return subg_dict, orig_nids, orig_eids
    else:
        return subg_dict, None, None


def get_peak_mem():
    """Get the peak memory size.

    Returns
    -------
    float
        The peak memory size in GB.
    """
    if not os.path.exists("/proc/self/status"):
        return 0.0
    for line in open("/proc/self/status", "r"):
        if "VmPeak" in line:
            mem = re.findall(r"\d+", line)[0]
            return int(mem) / 1024 / 1024
    return 0.0


def metis_partition_assignment(
    g, k, balance_ntypes=None, balance_edges=False, mode="k-way", objtype="cut"
):
    """This assigns nodes to different partitions with Metis partitioning algorithm.

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
    mode : str, "k-way" or "recursive"
        Whether use multilevel recursive bisection or multilevel k-way paritioning.
    objtype : str, "cut" or "vol"
        Set the objective as edge-cut minimization or communication volume minimization. This
        argument is used by the Metis algorithm.

    Returns
    -------
    a 1-D tensor
        A vector with each element that indicates the partition ID of a vertex.
    """
    assert mode in (
        "k-way",
        "recursive",
    ), "'mode' can only be 'k-way' or 'recursive'"
    assert (
        g.idtype == F.int64
    ), "IdType of graph is required to be int64 for now."
    # METIS works only on symmetric graphs.
    # The METIS runs on the symmetric graph to generate the node assignment to partitions.
    start = time.time()
    sym_gidx = _CAPI_DGLMakeSymmetric_Hetero(g._graph)
    sym_g = DGLGraph(gidx=sym_gidx)
    print(
        "Convert a graph into a bidirected graph: {:.3f} seconds, peak memory: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )
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
        assert (
            len(balance_ntypes) == g.num_nodes()
        ), "The length of balance_ntypes should be equal to #nodes in the graph"
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
                degs = np.zeros((g.num_nodes(),), np.int64)
                degs[nids] = F.asnumpy(g.in_degrees(nids))
                vwgt.append(F.zerocopy_from_numpy(degs))

    # The vertex weights have to be stored in a vector.
    if len(vwgt) > 0:
        vwgt = F.stack(vwgt, 1)
        shape = (
            np.prod(
                F.shape(vwgt),
            ),
        )
        vwgt = F.reshape(vwgt, shape)
        vwgt = F.to_dgl_nd(vwgt)
    else:
        vwgt = F.zeros((0,), F.int64, F.cpu())
        vwgt = F.to_dgl_nd(vwgt)
    print(
        "Construct multi-constraint weights: {:.3f} seconds, peak memory: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )

    start = time.time()
    node_part = _CAPI_DGLMetisPartition_Hetero(
        sym_g._graph, k, vwgt, mode, (objtype == "cut")
    )
    print(
        "Metis partitioning: {:.3f} seconds, peak memory: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )
    if len(node_part) == 0:
        return None
    else:
        node_part = utils.toindex(node_part)
        return node_part.tousertensor()


def metis_partition(
    g,
    k,
    extra_cached_hops=0,
    reshuffle=False,
    balance_ntypes=None,
    balance_edges=False,
    mode="k-way",
):
    """This is to partition a graph with Metis partitioning.

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

    If `reshuffle` is turned on, the function reshuffles node IDs and edge IDs
    of the input graph before partitioning. After reshuffling, all nodes and edges
    in a partition fall in a contiguous ID range in the input graph.
    The partitioend subgraphs have node data 'orig_id', which stores the node IDs
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
        Resuffle nodes so that nodes in the same partition are in the same ID range.
    balance_ntypes : tensor
        Node type of each node
    balance_edges : bool
        Indicate whether to balance the edges.
    mode : str, "k-way" or "recursive"
        Whether use multilevel recursive bisection or multilevel k-way paritioning.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition ID and the value is the DGLGraph of the partition.
    """
    assert mode in (
        "k-way",
        "recursive",
    ), "'mode' can only be 'k-way' or 'recursive'"
    node_part = metis_partition_assignment(
        g, k, balance_ntypes, balance_edges, mode
    )
    if node_part is None:
        return None

    # Then we split the original graph into parts based on the METIS partitioning results.
    return partition_graph_with_halo(
        g, node_part, extra_cached_hops, reshuffle
    )[0]


class NDArrayPartition(object):
    """Create a new partition of an NDArray. That is, an object which assigns
    each row of an NDArray to a specific partition.

    Parameters
    ----------
    array_size : int
        The first dimension of the array being partitioned.
    num_parts : int
        The number of parts to divide the array into.
    mode : String
        The type of partition. Currently, the only valid values are
        'remainder' and 'range'.
        'remainder' assigns rows based on remainder when dividing the row id by the
        number of parts (e.g., i % num_parts).
        'range' assigns rows based on which part of the range 'part_ranges'
        they fall into.
    part_ranges : Tensor or dgl.NDArray, Optional
        Should only be specified when the mode is 'range'. Should be of the
        length `num_parts + 1`, and be the exclusive prefix-sum of the number
        of nodes in each partition. That is, for 3 partitions, we could have
        the list [0, a, b, 'array_size'], and all rows with index less
        than 'a' are assigned to partition 0, all rows with index greater than
        or equal to 'a' and less than 'b' are in partition 1, and all rows
        with index greater or equal to 'b' are in partition 2. Should have
        the same context as the partitioned NDArray (i.e., be on the same GPU).

    Examples
    --------

    A partition of a homgeonous graph `g`, where the vertices are
    striped across processes can be generated via:

    >>> from dgl.partition import NDArrayPartition
    >>> part = NDArrayPartition(g.num_nodes(), num_parts, mode='remainder' )

    A range based partition of a homogenous graph `g`'s nodes, where
    the nodes are stored in contiguous memory. This converts an existing
    range based partitioning (e.g. from a
    dgl.distributed.graph_partition_book.RangePartitionBook)
    'max_node_map', to an NDArrayPartition 'part'.

    >>> part_range = [0]
    >>> for part in part_book.metadata():
    >>>     part_range.append(part_range[-1] + part['num_nodes'])
    >>> part = NDArrayPartition(g.num_nodes(), num_parts, mode='range',
    ...                         part_ranges=part_range)
    """

    def __init__(
        self, array_size, num_parts, mode="remainder", part_ranges=None
    ):
        assert num_parts > 0, 'Invalid "num_parts", must be > 0.'
        if mode == "remainder":
            assert part_ranges is None, (
                "When using remainder-based "
                'partitioning, "part_ranges" should not be specified.'
            )
            self._partition = _CAPI_DGLNDArrayPartitionCreateRemainderBased(
                array_size, num_parts
            )
        elif mode == "range":
            assert part_ranges is not None, (
                "When using range-based "
                'partitioning, "part_ranges" must not be None.'
            )
            assert part_ranges[0] == 0 and part_ranges[-1] == array_size, (
                "part_ranges[0] must be 0, and part_ranges[-1] must be "
                '"array_size".'
            )
            if F.is_tensor(part_ranges):
                part_ranges = F.zerocopy_to_dgl_ndarray(part_ranges)
            assert isinstance(part_ranges, NDArray), (
                '"part_ranges" must ' "be Tensor or dgl.NDArray."
            )
            self._partition = _CAPI_DGLNDArrayPartitionCreateRangeBased(
                array_size, num_parts, part_ranges
            )
        else:
            assert False, 'Unknown partition mode "{}"'.format(mode)
        self._array_size = array_size
        self._num_parts = num_parts

    def num_parts(self):
        """Get the number of partitions."""
        return self._num_parts

    def array_size(self):
        """Get the total size of the first dimension of the partitioned array."""
        return self._array_size

    def get(self):
        """Get the C-handle for this object."""
        return self._partition

    def get_local_indices(self, part, ctx):
        """Get the set of global indices in this given partition."""
        return self.map_to_global(
            F.arange(0, self.local_size(part), ctx=ctx), part
        )

    def local_size(self, part):
        """Get the number of rows/items assigned to the given part."""
        return _CAPI_DGLNDArrayPartitionGetPartSize(self._partition, part)

    def map_to_local(self, idxs):
        """Convert the set of global indices to local indices"""
        return F.zerocopy_from_dgl_ndarray(
            _CAPI_DGLNDArrayPartitionMapToLocal(
                self._partition, F.zerocopy_to_dgl_ndarray(idxs)
            )
        )

    def map_to_global(self, idxs, part_id):
        """Convert the set of local indices ot global indices"""
        return F.zerocopy_from_dgl_ndarray(
            _CAPI_DGLNDArrayPartitionMapToGlobal(
                self._partition, F.zerocopy_to_dgl_ndarray(idxs), part_id
            )
        )

    def generate_permutation(self, idxs):
        """Produce a scheme that maps the given indices to separate partitions
        and the counts of how many indices are in each partition.


        Parameters
        ----------
        idxs: torch.Tensor.
            A tensor with shape (`num_indices`,), representing global indices.

        Return
        ------
        torch.Tensor.
            A tensor with shape (`num_indices`,), representing the permutation
            to re-order the indices by partition.
        torch.Tensor.
            A tensor with shape (`num_partition`,), representing the number of
            indices per partition.

        Examples
        --------

        >>> import torch
        >>> from dgl.partition import NDArrayPartition
        >>> part = NDArrayPartition(10, 2, mode="remainder")
        >>> idx = torch.tensor([0, 2, 4, 5, 8, 8, 9], device="cuda:0")
        >>> perm, splits_sum = part.generate_permutation(idx)
        >>> perm
        tensor([0, 1, 2, 4, 5, 3, 6], device='cuda:0')
        >>> splits_sum
        tensor([5, 2], device='cuda:0')
        """
        ret = _CAPI_DGLNDArrayPartitionGeneratePermutation(
            self._partition, F.zerocopy_to_dgl_ndarray(idxs)
        )
        return F.zerocopy_from_dgl_ndarray(ret(0)), F.zerocopy_from_dgl_ndarray(
            ret(1)
        )


_init_api("dgl.partition")
