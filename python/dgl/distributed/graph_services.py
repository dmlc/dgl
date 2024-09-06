"""A set of graph services of getting subgraphs from DistGraph"""

import os
from collections import namedtuple

import numpy as np

import torch

from .. import backend as F, graphbolt as gb
from ..base import EID, ETYPE, NID
from ..convert import graph, heterograph
from ..sampling import (
    sample_etype_neighbors as local_sample_etype_neighbors,
    sample_neighbors as local_sample_neighbors,
)
from ..subgraph import in_subgraph as local_in_subgraph
from ..utils import toindex
from .constants import DGL2GB_EID, GB_DST_ID
from .rpc import (
    recv_responses,
    register_service,
    Request,
    Response,
    send_requests_to_machine,
)

__all__ = [
    "sample_neighbors",
    "sample_etype_neighbors",
    "in_subgraph",
    "find_edges",
]

SAMPLING_SERVICE_ID = 6657
INSUBGRAPH_SERVICE_ID = 6658
EDGES_SERVICE_ID = 6659
OUTDEGREE_SERVICE_ID = 6660
INDEGREE_SERVICE_ID = 6661
ETYPE_SAMPLING_SERVICE_ID = 6662


class SubgraphResponse(Response):
    """The response for sampling and in_subgraph"""

    def __init__(
        self, global_src, global_dst, *, global_eids=None, etype_ids=None
    ):
        self.global_src = global_src
        self.global_dst = global_dst
        self.global_eids = global_eids
        self.etype_ids = etype_ids

    def __setstate__(self, state):
        (
            self.global_src,
            self.global_dst,
            self.global_eids,
            self.etype_ids,
        ) = state

    def __getstate__(self):
        return (
            self.global_src,
            self.global_dst,
            self.global_eids,
            self.etype_ids,
        )


class FindEdgeResponse(Response):
    """The response for sampling and in_subgraph"""

    def __init__(self, global_src, global_dst, order_id):
        self.global_src = global_src
        self.global_dst = global_dst
        self.order_id = order_id

    def __setstate__(self, state):
        self.global_src, self.global_dst, self.order_id = state

    def __getstate__(self):
        return self.global_src, self.global_dst, self.order_id


def _sample_neighbors_graphbolt(
    g,
    gpb,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    exclude_edges=None,
    replace=False,
):
    """Sample from local partition via graphbolt.

    The input nodes use global IDs. We need to map the global node IDs to local
    node IDs, perform sampling and map the sampled results to the global IDs
    space again. The sampled results are stored in three vectors that store
    source nodes, destination nodes, etype IDs and edge IDs.

    Parameters
    ----------
    g : FusedCSCSamplingGraph
        The local partition.
    gpb : GraphPartitionBook
        The graph partition book.
    nodes : tensor
        The nodes to sample neighbors from.
    fanout : tensor or int
        The number of edges to be sampled for each node.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.
    prob : tensor, optional
        The probability associated with each neighboring edge of a node.
    exclude_edges : tensor, optional
        The edges to exclude when sampling.
    replace : bool, optional
        If True, sample with replacement.

    Returns
    -------
    tensor
        The source node ID array.
    tensor
        The destination node ID array.
    tensor
        The edge ID array.
    tensor
        The edge type ID array.
    """
    assert (
        edge_dir == "in"
    ), f"GraphBolt only supports inbound edge sampling but got {edge_dir}."
    assert exclude_edges is None, "GraphBolt does not support excluding edges."

    # 1. Map global node IDs to local node IDs.
    nodes = gpb.nid2localnid(nodes, gpb.partid)
    # Local partition may be saved in torch.int32 even though the global graph
    # is in torch.int64.
    nodes = nodes.to(dtype=g.indices.dtype)

    # 2. Perform sampling.
    probs_or_mask = None
    if prob is not None:
        probs_or_mask = g.edge_attributes[prob]
    # Sanity checks.
    assert isinstance(
        g, gb.FusedCSCSamplingGraph
    ), "Expect a FusedCSCSamplingGraph."
    assert isinstance(nodes, torch.Tensor), "Expect a tensor of nodes."
    if isinstance(fanout, int):
        fanout = torch.LongTensor([fanout])
    assert isinstance(fanout, torch.Tensor), "Expect a tensor of fanout."

    subgraph = g._sample_neighbors(
        nodes,
        None,
        fanout,
        replace=replace,
        probs_or_mask=probs_or_mask,
    )

    # 3. Map local node IDs to global node IDs.
    local_src = subgraph.indices
    local_dst = gb.expand_indptr(
        subgraph.indptr,
        dtype=local_src.dtype,
        node_ids=subgraph.original_column_node_ids,
        output_size=local_src.shape[0],
    )
    global_nid_mapping = g.node_attributes[NID]
    global_src = global_nid_mapping[local_src]
    global_dst = global_nid_mapping[local_dst]

    global_eids = None
    if g.edge_attributes is not None and EID in g.edge_attributes:
        global_eids = g.edge_attributes[EID][subgraph.original_edge_ids]
    return LocalSampledGraph(
        global_src, global_dst, global_eids, subgraph.type_per_edge
    )


def _sample_neighbors_dgl(
    local_g,
    partition_book,
    seed_nodes,
    fan_out,
    edge_dir="in",
    prob=None,
    exclude_edges=None,
    replace=False,
):
    """Sample from local partition.

    The input nodes use global IDs. We need to map the global node IDs to local node IDs,
    perform sampling and map the sampled results to the global IDs space again.
    The sampled results are stored in three vectors that store source nodes, destination nodes
    and edge IDs.
    """
    local_ids = partition_book.nid2localnid(seed_nodes, partition_book.partid)
    local_ids = F.astype(local_ids, local_g.idtype)
    # local_ids = self.seed_nodes
    sampled_graph = local_sample_neighbors(
        local_g,
        local_ids,
        fan_out,
        edge_dir=edge_dir,
        prob=prob,
        exclude_edges=exclude_edges,
        replace=replace,
        _dist_training=True,
    )
    global_nid_mapping = local_g.ndata[NID]
    src, dst = sampled_graph.edges()
    global_src, global_dst = F.gather_row(
        global_nid_mapping, src
    ), F.gather_row(global_nid_mapping, dst)
    global_eids = F.gather_row(local_g.edata[EID], sampled_graph.edata[EID])
    return LocalSampledGraph(global_src, global_dst, global_eids)


def _sample_neighbors(use_graphbolt, *args, **kwargs):
    """Wrapper for sampling neighbors.

    The actual sampling function depends on whether to use GraphBolt.

    Parameters
    ----------
    use_graphbolt : bool
        Whether to use GraphBolt for sampling.
    args : list
        The arguments for the sampling function.
    kwargs : dict
        The keyword arguments for the sampling function.

    Returns
    -------
    tensor
        The source node ID array.
    tensor
        The destination node ID array.
    tensor
        The edge ID array.
    tensor
        The edge type ID array.
    """
    func = (
        _sample_neighbors_graphbolt if use_graphbolt else _sample_neighbors_dgl
    )
    return func(*args, **kwargs)


def _sample_etype_neighbors_dgl(
    local_g,
    partition_book,
    seed_nodes,
    fan_out,
    edge_dir="in",
    prob=None,
    exclude_edges=None,
    replace=False,
    etype_offset=None,
    etype_sorted=False,
):
    """Sample from local partition.

    The input nodes use global IDs. We need to map the global node IDs to local node IDs,
    perform sampling and map the sampled results to the global IDs space again.
    The sampled results are stored in three vectors that store source nodes, destination nodes
    and edge IDs.
    """
    assert etype_offset is not None, "The etype offset is not provided."

    local_ids = partition_book.nid2localnid(seed_nodes, partition_book.partid)
    local_ids = F.astype(local_ids, local_g.idtype)

    sampled_graph = local_sample_etype_neighbors(
        local_g,
        local_ids,
        etype_offset,
        fan_out,
        edge_dir=edge_dir,
        prob=prob,
        exclude_edges=exclude_edges,
        replace=replace,
        etype_sorted=etype_sorted,
        _dist_training=True,
    )
    global_nid_mapping = local_g.ndata[NID]
    src, dst = sampled_graph.edges()
    global_src, global_dst = F.gather_row(
        global_nid_mapping, src
    ), F.gather_row(global_nid_mapping, dst)
    global_eids = F.gather_row(local_g.edata[EID], sampled_graph.edata[EID])
    return LocalSampledGraph(global_src, global_dst, global_eids)


def _sample_etype_neighbors(use_graphbolt, *args, **kwargs):
    """Wrapper for sampling etype neighbors.

    The actual sampling function depends on whether to use GraphBolt.

    Parameters
    ----------
    use_graphbolt : bool
        Whether to use GraphBolt for sampling.
    args : list
        The arguments for the sampling function.
    kwargs : dict
        The keyword arguments for the sampling function.

    Returns
    -------
    tensor
        The source node ID array.
    tensor
        The destination node ID array.
    tensor
        The edge ID array.
    tensor
        The edge type ID array.
    """
    func = (
        _sample_neighbors_graphbolt
        if use_graphbolt
        else _sample_etype_neighbors_dgl
    )
    if use_graphbolt:
        # GraphBolt does not require `etype_offset` and `etype_sorted`.
        kwargs.pop("etype_offset", None)
        kwargs.pop("etype_sorted", None)
    return func(*args, **kwargs)


def _find_edges(local_g, partition_book, seed_edges):
    """Given an edge ID array, return the source
    and destination node ID array ``s`` and ``d`` in the local partition.
    """
    local_eids = partition_book.eid2localeid(seed_edges, partition_book.partid)
    if isinstance(local_g, gb.FusedCSCSamplingGraph):
        # When converting from DGLGraph to FusedCSCSamplingGraph, the edge IDs
        # are re-ordered. In order to find the correct node pairs, we need to
        # map the DGL edge IDs back to GraphBolt edge IDs.
        if (
            DGL2GB_EID not in local_g.edge_attributes
            or GB_DST_ID not in local_g.edge_attributes
        ):
            raise ValueError(
                "The edge attributes DGL2GB_EID and GB_DST_ID are not found. "
                "Please make sure `coo` format is available when generating "
                "partitions in GraphBolt format."
            )
        local_eids = local_g.edge_attributes[DGL2GB_EID][local_eids]
        local_src = local_g.indices[local_eids]
        local_dst = local_g.edge_attributes[GB_DST_ID][local_eids]
        global_nid_mapping = local_g.node_attributes[NID]
    else:
        local_eids = F.astype(local_eids, local_g.idtype)
        local_src, local_dst = local_g.find_edges(local_eids)
        global_nid_mapping = local_g.ndata[NID]
    global_src = global_nid_mapping[local_src]
    global_dst = global_nid_mapping[local_dst]
    return global_src, global_dst


def _in_degrees(local_g, partition_book, n):
    """Get in-degree of the nodes in the local partition."""
    local_nids = partition_book.nid2localnid(n, partition_book.partid)
    local_nids = F.astype(local_nids, local_g.idtype)
    return local_g.in_degrees(local_nids)


def _out_degrees(local_g, partition_book, n):
    """Get out-degree of the nodes in the local partition."""
    local_nids = partition_book.nid2localnid(n, partition_book.partid)
    local_nids = F.astype(local_nids, local_g.idtype)
    return local_g.out_degrees(local_nids)


def _in_subgraph(local_g, partition_book, seed_nodes):
    """Get in subgraph from local partition.

    The input nodes use global IDs. We need to map the global node IDs to local node IDs,
    get in-subgraph and map the sampled results to the global IDs space again.
    The results are stored in three vectors that store source nodes, destination nodes
    and edge IDs.
    """
    local_ids = partition_book.nid2localnid(seed_nodes, partition_book.partid)
    local_ids = F.astype(local_ids, local_g.idtype)
    # local_ids = self.seed_nodes
    sampled_graph = local_in_subgraph(local_g, local_ids)
    global_nid_mapping = local_g.ndata[NID]
    src, dst = sampled_graph.edges()
    global_src, global_dst = global_nid_mapping[src], global_nid_mapping[dst]
    global_eids = F.gather_row(local_g.edata[EID], sampled_graph.edata[EID])
    return LocalSampledGraph(global_src, global_dst, global_eids)


# --- NOTE 1 ---
# (BarclayII)
# If the sampling algorithm needs node and edge data, ideally the
# algorithm should query the underlying feature storage to get what it
# just needs to complete the job.  For instance, with
# sample_etype_neighbors, we only need the probability of the seed nodes'
# neighbors.
#
# However, right now we are reusing the existing subgraph sampling
# interfaces of DGLGraph (i.e. single machine solution), which needs
# the data of *all* the nodes/edges.  Going distributed, we now need
# the node/edge data of the *entire* local graph partition.
#
# If the sampling algorithm only use edge data, the current design works
# because the local graph partition contains all the in-edges of the
# assigned nodes as well as the data.  This is the case for
# sample_etype_neighbors.
#
# However, if the sampling algorithm requires data of the neighbor nodes
# (e.g. sample_neighbors_biased which performs biased sampling based on the
# type of the neighbor nodes), the current design will fail because the
# neighbor nodes (hence the data) may not belong to the current partition.
# This is a limitation of the current DistDGL design.  We should improve it
# later.


class SamplingRequest(Request):
    """Sampling Request"""

    def __init__(
        self,
        nodes,
        fan_out,
        edge_dir="in",
        prob=None,
        exclude_edges=None,
        replace=False,
        use_graphbolt=False,
    ):
        self.seed_nodes = nodes
        self.edge_dir = edge_dir
        self.prob = prob
        self.exclude_edges = exclude_edges
        self.replace = replace
        self.fan_out = fan_out
        self.use_graphbolt = use_graphbolt

    def __setstate__(self, state):
        (
            self.seed_nodes,
            self.edge_dir,
            self.prob,
            self.exclude_edges,
            self.replace,
            self.fan_out,
            self.use_graphbolt,
        ) = state

    def __getstate__(self):
        return (
            self.seed_nodes,
            self.edge_dir,
            self.prob,
            self.exclude_edges,
            self.replace,
            self.fan_out,
            self.use_graphbolt,
        )

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        kv_store = server_state.kv_store
        if self.prob is not None and (not self.use_graphbolt):
            prob = [kv_store.data_store[self.prob]]
        else:
            prob = self.prob
        res = _sample_neighbors(
            self.use_graphbolt,
            local_g,
            partition_book,
            self.seed_nodes,
            self.fan_out,
            edge_dir=self.edge_dir,
            prob=prob,
            exclude_edges=self.exclude_edges,
            replace=self.replace,
        )
        return SubgraphResponse(
            res.global_src,
            res.global_dst,
            global_eids=res.global_eids,
            etype_ids=res.etype_ids,
        )


class SamplingRequestEtype(Request):
    """Sampling Request"""

    def __init__(
        self,
        nodes,
        fan_out,
        edge_dir="in",
        prob=None,
        exclude_edges=None,
        replace=False,
        etype_sorted=True,
        use_graphbolt=False,
    ):
        self.seed_nodes = nodes
        self.edge_dir = edge_dir
        self.prob = prob
        self.exclude_edges = exclude_edges
        self.replace = replace
        self.fan_out = fan_out
        self.etype_sorted = etype_sorted
        self.use_graphbolt = use_graphbolt

    def __setstate__(self, state):
        (
            self.seed_nodes,
            self.edge_dir,
            self.prob,
            self.exclude_edges,
            self.replace,
            self.fan_out,
            self.etype_sorted,
            self.use_graphbolt,
        ) = state

    def __getstate__(self):
        return (
            self.seed_nodes,
            self.edge_dir,
            self.prob,
            self.exclude_edges,
            self.replace,
            self.fan_out,
            self.etype_sorted,
            self.use_graphbolt,
        )

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        kv_store = server_state.kv_store
        etype_offset = partition_book.local_etype_offset
        # See NOTE 1
        if self.prob is not None and (not self.use_graphbolt):
            probs = [
                kv_store.data_store[key] if key != "" else None
                for key in self.prob
            ]
        else:
            probs = self.prob
        res = _sample_etype_neighbors(
            self.use_graphbolt,
            local_g,
            partition_book,
            self.seed_nodes,
            self.fan_out,
            edge_dir=self.edge_dir,
            prob=probs,
            exclude_edges=self.exclude_edges,
            replace=self.replace,
            etype_offset=etype_offset,
            etype_sorted=self.etype_sorted,
        )
        return SubgraphResponse(
            res.global_src,
            res.global_dst,
            global_eids=res.global_eids,
            etype_ids=res.etype_ids,
        )


class EdgesRequest(Request):
    """Edges Request"""

    def __init__(self, edge_ids, order_id):
        self.edge_ids = edge_ids
        self.order_id = order_id

    def __setstate__(self, state):
        self.edge_ids, self.order_id = state

    def __getstate__(self):
        return self.edge_ids, self.order_id

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        global_src, global_dst = _find_edges(
            local_g, partition_book, self.edge_ids
        )

        return FindEdgeResponse(global_src, global_dst, self.order_id)


class InDegreeRequest(Request):
    """In-degree Request"""

    def __init__(self, n, order_id):
        self.n = n
        self.order_id = order_id

    def __setstate__(self, state):
        self.n, self.order_id = state

    def __getstate__(self):
        return self.n, self.order_id

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        deg = _in_degrees(local_g, partition_book, self.n)

        return InDegreeResponse(deg, self.order_id)


class InDegreeResponse(Response):
    """The response for in-degree"""

    def __init__(self, deg, order_id):
        self.val = deg
        self.order_id = order_id

    def __setstate__(self, state):
        self.val, self.order_id = state

    def __getstate__(self):
        return self.val, self.order_id


class OutDegreeRequest(Request):
    """Out-degree Request"""

    def __init__(self, n, order_id):
        self.n = n
        self.order_id = order_id

    def __setstate__(self, state):
        self.n, self.order_id = state

    def __getstate__(self):
        return self.n, self.order_id

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        deg = _out_degrees(local_g, partition_book, self.n)

        return OutDegreeResponse(deg, self.order_id)


class OutDegreeResponse(Response):
    """The response for out-degree"""

    def __init__(self, deg, order_id):
        self.val = deg
        self.order_id = order_id

    def __setstate__(self, state):
        self.val, self.order_id = state

    def __getstate__(self):
        return self.val, self.order_id


class InSubgraphRequest(Request):
    """InSubgraph Request"""

    def __init__(self, nodes):
        self.seed_nodes = nodes

    def __setstate__(self, state):
        self.seed_nodes = state

    def __getstate__(self):
        return self.seed_nodes

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        global_src, global_dst, global_eids = _in_subgraph(
            local_g, partition_book, self.seed_nodes
        )
        return SubgraphResponse(global_src, global_dst, global_eids=global_eids)


def merge_graphs(res_list, num_nodes, exclude_edges=None):
    """Merge request from multiple servers"""
    if len(res_list) > 1:
        srcs = []
        dsts = []
        eids = []
        etype_ids = []
        for res in res_list:
            srcs.append(res.global_src)
            dsts.append(res.global_dst)
            eids.append(res.global_eids)
            etype_ids.append(res.etype_ids)
        src_tensor = F.cat(srcs, 0)
        dst_tensor = F.cat(dsts, 0)
        eid_tensor = None if eids[0] is None else F.cat(eids, 0)
        etype_id_tensor = None if etype_ids[0] is None else F.cat(etype_ids, 0)
    else:
        src_tensor = res_list[0].global_src
        dst_tensor = res_list[0].global_dst
        eid_tensor = res_list[0].global_eids
        etype_id_tensor = res_list[0].etype_ids
    if exclude_edges is not None:
        mask = torch.isin(
            eid_tensor, exclude_edges, assume_unique=True, invert=True
        )
        src_tensor = src_tensor[mask]
        dst_tensor = dst_tensor[mask]
        eid_tensor = eid_tensor[mask]
        if etype_id_tensor is not None:
            etype_id_tensor = etype_id_tensor[mask]
    g = graph((src_tensor, dst_tensor), num_nodes=num_nodes)
    if eid_tensor is not None:
        g.edata[EID] = eid_tensor
    if etype_id_tensor is not None:
        g.edata[ETYPE] = etype_id_tensor
    return g


LocalSampledGraph = namedtuple(  # pylint: disable=unexpected-keyword-arg
    "LocalSampledGraph",
    "global_src global_dst global_eids etype_ids",
    defaults=(None, None, None, None),
)


def _distributed_access(
    g, nodes, issue_remote_req, local_access, exclude_edges=None
):
    """A routine that fetches local neighborhood of nodes from the distributed graph.

    The local neighborhood of some nodes are stored in the local machine and the other
    nodes have their neighborhood on remote machines. This code will issue remote
    access requests first before fetching data from the local machine. In the end,
    we combine the data from the local machine and remote machines.
    In this way, we can hide the latency of accessing data on remote machines.

    Parameters
    ----------
    g : DistGraph
        The distributed graph
    nodes : tensor
        The nodes whose neighborhood are to be fetched.
    issue_remote_req : callable
        The function that issues requests to access remote data.
    local_access : callable
        The function that reads data on the local machine.
    exclude_edges : tensor
        The edges to exclude after sampling.

    Returns
    -------
    DGLGraph
        The subgraph that contains the neighborhoods of all input nodes.
    """
    req_list = []
    partition_book = g.get_partition_book()
    if not isinstance(nodes, torch.Tensor):
        nodes = toindex(nodes).tousertensor()
    partition_id = partition_book.nid2partid(nodes)
    local_nids = None
    for pid in range(partition_book.num_partitions()):
        node_id = F.boolean_mask(nodes, partition_id == pid)
        # We optimize the sampling on a local partition if the server and the client
        # run on the same machine. With a good partitioning, most of the seed nodes
        # should reside in the local partition. If the server and the client
        # are not co-located, the client doesn't have a local partition.
        if pid == partition_book.partid and g.local_partition is not None:
            assert local_nids is None
            local_nids = node_id
        elif len(node_id) != 0:
            req = issue_remote_req(node_id)
            req_list.append((pid, req))

    # send requests to the remote machine.
    msgseq2pos = None
    if len(req_list) > 0:
        msgseq2pos = send_requests_to_machine(req_list)

    # sample neighbors for the nodes in the local partition.
    res_list = []
    if local_nids is not None:
        res = local_access(g.local_partition, partition_book, local_nids)
        res_list.append(res)

    # receive responses from remote machines.
    if msgseq2pos is not None:
        results = recv_responses(msgseq2pos)
        res_list.extend(results)

    sampled_graph = merge_graphs(
        res_list, g.num_nodes(), exclude_edges=exclude_edges
    )
    return sampled_graph


def _frontier_to_heterogeneous_graph(g, frontier, gpb):
    # We need to handle empty frontiers correctly.
    if frontier.num_edges() == 0:
        data_dict = {
            etype: (np.zeros(0), np.zeros(0)) for etype in g.canonical_etypes
        }
        return heterograph(
            data_dict,
            {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
            idtype=g.idtype,
        )

    # For DGL partitions, the global edge IDs are always stored in the edata.
    # For GraphBolt partitions, the edge type IDs are always stored in the
    # edata. As for the edge IDs, they are stored in the edata if the graph is
    # partitioned with `store_eids=True`. Otherwise, the edge IDs are not
    # stored.
    etype_ids, type_wise_eids = (
        gpb.map_to_per_etype(frontier.edata[EID])
        if EID in frontier.edata
        else (frontier.edata[ETYPE], None)
    )
    etype_ids, idx = F.sort_1d(etype_ids)
    if type_wise_eids is not None:
        type_wise_eids = F.gather_row(type_wise_eids, idx)

    # Sort the edges by their edge types.
    src, dst = frontier.edges()
    src, dst = F.gather_row(src, idx), F.gather_row(dst, idx)
    src_ntype_ids, src = gpb.map_to_per_ntype(src)
    dst_ntype_ids, dst = gpb.map_to_per_ntype(dst)

    data_dict = dict()
    edge_ids = {}
    for etid, etype in enumerate(g.canonical_etypes):
        src_ntype, _, dst_ntype = etype
        src_ntype_id = g.get_ntype_id(src_ntype)
        dst_ntype_id = g.get_ntype_id(dst_ntype)
        type_idx = etype_ids == etid
        data_dict[etype] = (
            F.boolean_mask(src, type_idx),
            F.boolean_mask(dst, type_idx),
        )
        if "DGL_DIST_DEBUG" in os.environ:
            assert torch.all(
                src_ntype_id == src_ntype_ids[type_idx]
            ), "source ntype is is not expected."
            assert torch.all(
                dst_ntype_id == dst_ntype_ids[type_idx]
            ), "destination ntype is is not expected."
        if type_wise_eids is not None:
            edge_ids[etype] = F.boolean_mask(type_wise_eids, type_idx)
    hg = heterograph(
        data_dict,
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        idtype=g.idtype,
    )

    for etype in edge_ids:
        hg.edges[etype].data[EID] = edge_ids[etype]
    return hg


def sample_etype_neighbors(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    exclude_edges=None,
    replace=False,
    etype_sorted=True,
    use_graphbolt=False,
):
    """Sample from the neighbors of the given nodes from a distributed graph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The returned graph will contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    This function assumes the input is a homogeneous ``DGLGraph`` with the edges
    ordered by their edge types. The sampled subgraph is also
    stored in the homogeneous graph format. That is, all nodes and edges are assigned
    with unique IDs (in contrast, we typically use a type name and a node/edge ID to
    identify a node or an edge in ``DGLGraph``). We refer to this type of IDs
    as *homogeneous ID*.
    Users can use :func:`dgl.distributed.GraphPartitionBook.map_to_per_ntype`
    and :func:`dgl.distributed.GraphPartitionBook.map_to_per_etype`
    to identify their node/edge types and node/edge IDs of that type.

    Parameters
    ----------
    g : DistGraph
        The distributed graph..
    nodes : tensor or dict
        Node IDs to sample neighbors from. If it's a dict, it should contain only
        one key-value pair to make this API consistent with dgl.sampling.sample_neighbors.
    fanout : int or dict[etype, int]
        The number of edges to be sampled for each node per edge type.  If an integer
        is given, DGL assumes that the same fanout is applied to every edge type.

        If -1 is given, all of the neighbors will be selected.
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
    exclude_edges : tensor, optional
        The edges to exclude when sampling. Homogeneous edge IDs are used.
    replace : bool, optional
        If True, sample with replacement.

        When sampling with replacement, the sampled subgraph could have parallel edges.

        For sampling without replacement, if fanout > the number of neighbors, all the
        neighbors are sampled. If fanout == -1, all neighbors are collected.
    etype_sorted : bool, optional
        Indicates whether etypes are sorted.
    use_graphbolt : bool, optional
        Whether to use GraphBolt for sampling.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.  It is on CPU.
    """
    if isinstance(fanout, int):
        fanout = F.full_1d(len(g.canonical_etypes), fanout, F.int64, F.cpu())
    else:
        etype_ids = {etype: i for i, etype in enumerate(g.canonical_etypes)}
        fanout_array = [None] * len(g.canonical_etypes)
        for etype, v in fanout.items():
            c_etype = g.to_canonical_etype(etype)
            fanout_array[etype_ids[c_etype]] = v
        assert all(v is not None for v in fanout_array), (
            "Not all etypes have valid fanout. Please make sure passed-in "
            "fanout in dict includes all the etypes in graph. Passed-in "
            f"fanout: {fanout}, graph etypes: {g.canonical_etypes}."
        )
        fanout = F.tensor(fanout_array, dtype=F.int64)

    gpb = g.get_partition_book()
    if isinstance(nodes, dict):
        homo_nids = []
        for ntype in nodes.keys():
            assert (
                ntype in g.ntypes
            ), "The sampled node type {} does not exist in the input graph".format(
                ntype
            )
            if F.is_tensor(nodes[ntype]):
                typed_nodes = nodes[ntype]
            else:
                typed_nodes = toindex(nodes[ntype]).tousertensor()
            homo_nids.append(gpb.map_to_homo_nid(typed_nodes, ntype))
        nodes = F.cat(homo_nids, 0)

    def issue_remote_req(node_ids):
        if prob is not None and (not use_graphbolt):
            # See NOTE 1
            _prob = [
                (
                    # NOTE (BarclayII)
                    # Currently DistGraph.edges[] does not accept canonical etype.
                    g.edges[etype].data[prob].kvstore_key
                    if prob in g.edges[etype].data
                    else ""
                )
                for etype in g.canonical_etypes
            ]
        else:
            _prob = prob
        return SamplingRequestEtype(
            node_ids,
            fanout,
            edge_dir=edge_dir,
            prob=_prob,
            exclude_edges=None,
            replace=replace,
            etype_sorted=etype_sorted,
            use_graphbolt=use_graphbolt,
        )

    def local_access(local_g, partition_book, local_nids):
        etype_offset = gpb.local_etype_offset
        # See NOTE 1
        if prob is not None and (not use_graphbolt):
            _prob = [
                (
                    g.edges[etype].data[prob].local_partition
                    if prob in g.edges[etype].data
                    else None
                )
                for etype in g.canonical_etypes
            ]
        else:
            _prob = prob
        return _sample_etype_neighbors(
            use_graphbolt,
            local_g,
            partition_book,
            local_nids,
            fanout,
            edge_dir=edge_dir,
            prob=_prob,
            exclude_edges=None,
            replace=replace,
            etype_offset=etype_offset,
            etype_sorted=etype_sorted,
        )

    frontier = _distributed_access(
        g, nodes, issue_remote_req, local_access, exclude_edges=exclude_edges
    )
    if not gpb.is_homogeneous:
        return _frontier_to_heterogeneous_graph(g, frontier, gpb)
    else:
        return frontier


def sample_neighbors(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    exclude_edges=None,
    replace=False,
    use_graphbolt=False,
):
    """Sample from the neighbors of the given nodes from a distributed graph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The returned graph will contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    For heterogeneous graphs, ``nodes`` is a dictionary whose key is node type
    and the value is type-specific node IDs.

    Parameters
    ----------
    g : DistGraph
        The distributed graph..
    nodes : tensor or dict
        Node IDs to sample neighbors from. If it's a dict, it should contain only
        one key-value pair to make this API consistent with dgl.sampling.sample_neighbors.
    fanout : int
        The number of edges to be sampled for each node.

        If -1 is given, all of the neighbors will be selected.
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
    exclude_edges: tensor or dict, optional
        Edge IDs to exclude during sampling neighbors for the seed nodes.

        This argument can take a single ID tensor or a dictionary of edge types
        and ID tensors. If a single tensor is given, the graph must only have
        one type of nodes.
    replace : bool, optional
        If True, sample with replacement.

        When sampling with replacement, the sampled subgraph could have parallel edges.

        For sampling without replacement, if fanout > the number of neighbors, all the
        neighbors are sampled. If fanout == -1, all neighbors are collected.
    use_graphbolt : bool, optional
        Whether to use GraphBolt for sampling.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.  It is on CPU.
    """
    gpb = g.get_partition_book()
    if not gpb.is_homogeneous:
        assert isinstance(nodes, dict)
        homo_nids = []
        for ntype in nodes:
            assert (
                ntype in g.ntypes
            ), "The sampled node type does not exist in the input graph"
            if F.is_tensor(nodes[ntype]):
                typed_nodes = nodes[ntype]
            else:
                typed_nodes = toindex(nodes[ntype]).tousertensor()
            homo_nids.append(gpb.map_to_homo_nid(typed_nodes, ntype))
        nodes = F.cat(homo_nids, 0)
    elif isinstance(nodes, dict):
        assert len(nodes) == 1
        nodes = list(nodes.values())[0]

    def issue_remote_req(node_ids):
        if prob is not None and (not use_graphbolt):
            # See NOTE 1
            _prob = g.edata[prob].kvstore_key
        else:
            _prob = prob
        return SamplingRequest(
            node_ids,
            fanout,
            edge_dir=edge_dir,
            prob=_prob,
            exclude_edges=None,
            replace=replace,
            use_graphbolt=use_graphbolt,
        )

    def local_access(local_g, partition_book, local_nids):
        # See NOTE 1
        _prob = (
            [g.edata[prob].local_partition]
            if prob is not None and (not use_graphbolt)
            else prob
        )
        return _sample_neighbors(
            use_graphbolt,
            local_g,
            partition_book,
            local_nids,
            fanout,
            edge_dir=edge_dir,
            prob=_prob,
            exclude_edges=None,
            replace=replace,
        )

    frontier = _distributed_access(
        g, nodes, issue_remote_req, local_access, exclude_edges=exclude_edges
    )
    if not gpb.is_homogeneous:
        return _frontier_to_heterogeneous_graph(g, frontier, gpb)
    else:
        return frontier


def _distributed_edge_access(g, edges, issue_remote_req, local_access):
    """A routine that fetches local edges from distributed graph.

    The source and destination nodes of local edges are stored in the local
    machine and others are stored on remote machines. This code will issue
    remote access requests first before fetching data from the local machine.
    In the end, we combine the data from the local machine and remote machines.

    Parameters
    ----------
    g : DistGraph
        The distributed graph
    edges : tensor
        The edges to find their source and destination nodes.
    issue_remote_req : callable
        The function that issues requests to access remote data.
    local_access : callable
        The function that reads data on the local machine.

    Returns
    -------
    tensor
        The source node ID array.
    tensor
        The destination node ID array.
    """
    req_list = []
    partition_book = g.get_partition_book()
    edges = toindex(edges).tousertensor()
    partition_id = partition_book.eid2partid(edges)
    local_eids = None
    reorder_idx = []
    for pid in range(partition_book.num_partitions()):
        mask = partition_id == pid
        edge_id = F.boolean_mask(edges, mask)
        reorder_idx.append(F.nonzero_1d(mask))
        if pid == partition_book.partid and g.local_partition is not None:
            assert local_eids is None
            local_eids = edge_id
        elif len(edge_id) != 0:
            req = issue_remote_req(edge_id, pid)
            req_list.append((pid, req))

    # send requests to the remote machine.
    msgseq2pos = None
    if len(req_list) > 0:
        msgseq2pos = send_requests_to_machine(req_list)

    # handle edges in local partition.
    src_ids = F.zeros_like(edges)
    dst_ids = F.zeros_like(edges)
    if local_eids is not None:
        src, dst = local_access(g.local_partition, partition_book, local_eids)
        src_ids = F.scatter_row(
            src_ids, reorder_idx[partition_book.partid], src
        )
        dst_ids = F.scatter_row(
            dst_ids, reorder_idx[partition_book.partid], dst
        )

    # receive responses from remote machines.
    if msgseq2pos is not None:
        results = recv_responses(msgseq2pos)
        for result in results:
            src = result.global_src
            dst = result.global_dst
            src_ids = F.scatter_row(src_ids, reorder_idx[result.order_id], src)
            dst_ids = F.scatter_row(dst_ids, reorder_idx[result.order_id], dst)
    return src_ids, dst_ids


def find_edges(g, edge_ids):
    """Given an edge ID array, return the source and destination
    node ID array ``s`` and ``d`` from a distributed graph.
    ``s[i]`` and ``d[i]`` are source and destination node ID for
    edge ``eid[i]``.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    edges : tensor
        The edge ID array.

    Returns
    -------
    tensor
        The source node ID array.
    tensor
        The destination node ID array.
    """

    def issue_remote_req(edge_ids, order_id):
        return EdgesRequest(edge_ids, order_id)

    def local_access(local_g, partition_book, edge_ids):
        return _find_edges(local_g, partition_book, edge_ids)

    return _distributed_edge_access(g, edge_ids, issue_remote_req, local_access)


def in_subgraph(g, nodes):
    """Return the subgraph induced on the inbound edges of the given nodes.

    The subgraph keeps the same type schema and all the nodes are preserved regardless
    of whether they have an edge or not.

    Node/edge features are not preserved. The original IDs of
    the extracted edges are stored as the `dgl.EID` feature in the returned graph.

    For now, we only support the input graph with one node type and one edge type.


    Parameters
    ----------
    g : DistGraph
        The distributed graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from.

    Returns
    -------
    DGLGraph
        The subgraph.

        One can retrieve the mapping from subgraph edge ID to parent
        edge ID via ``dgl.EID`` edge features of the subgraph.
    """
    if isinstance(nodes, dict):
        assert (
            len(nodes) == 1
        ), "The distributed in_subgraph only supports one node type for now."
        nodes = list(nodes.values())[0]

    def issue_remote_req(node_ids):
        return InSubgraphRequest(node_ids)

    def local_access(local_g, partition_book, local_nids):
        return _in_subgraph(local_g, partition_book, local_nids)

    return _distributed_access(g, nodes, issue_remote_req, local_access)


def _distributed_get_node_property(g, n, issue_remote_req, local_access):
    req_list = []
    partition_book = g.get_partition_book()
    n = toindex(n).tousertensor()
    partition_id = partition_book.nid2partid(n)
    local_nids = None
    reorder_idx = []
    for pid in range(partition_book.num_partitions()):
        mask = partition_id == pid
        nid = F.boolean_mask(n, mask)
        reorder_idx.append(F.nonzero_1d(mask))
        if pid == partition_book.partid and g.local_partition is not None:
            assert local_nids is None
            local_nids = nid
        elif len(nid) != 0:
            req = issue_remote_req(nid, pid)
            req_list.append((pid, req))

    # send requests to the remote machine.
    msgseq2pos = None
    if len(req_list) > 0:
        msgseq2pos = send_requests_to_machine(req_list)

    # handle edges in local partition.
    vals = None
    if local_nids is not None:
        local_vals = local_access(g.local_partition, partition_book, local_nids)
        shape = list(F.shape(local_vals))
        shape[0] = len(n)
        vals = F.zeros(shape, F.dtype(local_vals), F.cpu())
        vals = F.scatter_row(
            vals, reorder_idx[partition_book.partid], local_vals
        )

    # receive responses from remote machines.
    if msgseq2pos is not None:
        results = recv_responses(msgseq2pos)
        if len(results) > 0 and vals is None:
            shape = list(F.shape(results[0].val))
            shape[0] = len(n)
            vals = F.zeros(shape, F.dtype(results[0].val), F.cpu())
        for result in results:
            val = result.val
            vals = F.scatter_row(vals, reorder_idx[result.order_id], val)
    return vals


def in_degrees(g, v):
    """Get in-degrees"""

    def issue_remote_req(v, order_id):
        return InDegreeRequest(v, order_id)

    def local_access(local_g, partition_book, v):
        return _in_degrees(local_g, partition_book, v)

    return _distributed_get_node_property(g, v, issue_remote_req, local_access)


def out_degrees(g, u):
    """Get out-degrees"""

    def issue_remote_req(u, order_id):
        return OutDegreeRequest(u, order_id)

    def local_access(local_g, partition_book, u):
        return _out_degrees(local_g, partition_book, u)

    return _distributed_get_node_property(g, u, issue_remote_req, local_access)


register_service(SAMPLING_SERVICE_ID, SamplingRequest, SubgraphResponse)
register_service(EDGES_SERVICE_ID, EdgesRequest, FindEdgeResponse)
register_service(INSUBGRAPH_SERVICE_ID, InSubgraphRequest, SubgraphResponse)
register_service(OUTDEGREE_SERVICE_ID, OutDegreeRequest, OutDegreeResponse)
register_service(INDEGREE_SERVICE_ID, InDegreeRequest, InDegreeResponse)
register_service(
    ETYPE_SAMPLING_SERVICE_ID, SamplingRequestEtype, SubgraphResponse
)
