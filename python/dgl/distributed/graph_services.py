"""A set of graph services of getting subgraphs from DistGraph"""
from collections import namedtuple

from .rpc import Request, Response, send_requests_to_machine, recv_responses
from ..sampling import sample_neighbors as local_sample_neighbors
from ..subgraph import in_subgraph as local_in_subgraph
from .rpc import register_service
from ..convert import graph
from ..base import NID, EID
from ..utils import toindex
from .. import backend as F

__all__ = ['sample_neighbors', 'in_subgraph', 'find_edges']

SAMPLING_SERVICE_ID = 6657
INSUBGRAPH_SERVICE_ID = 6658
EDGES_SERVICE_ID = 6659

class SubgraphResponse(Response):
    """The response for sampling and in_subgraph"""

    def __init__(self, global_src, global_dst, global_eids):
        self.global_src = global_src
        self.global_dst = global_dst
        self.global_eids = global_eids

    def __setstate__(self, state):
        self.global_src, self.global_dst, self.global_eids = state

    def __getstate__(self):
        return self.global_src, self.global_dst, self.global_eids

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

def _sample_neighbors(local_g, partition_book, seed_nodes, fan_out, edge_dir, prob, replace):
    """ Sample from local partition.

    The input nodes use global IDs. We need to map the global node IDs to local node IDs,
    perform sampling and map the sampled results to the global IDs space again.
    The sampled results are stored in three vectors that store source nodes, destination nodes
    and edge IDs.
    """
    local_ids = partition_book.nid2localnid(seed_nodes, partition_book.partid)
    local_ids = F.astype(local_ids, local_g.idtype)
    # local_ids = self.seed_nodes
    sampled_graph = local_sample_neighbors(
        local_g, local_ids, fan_out, edge_dir, prob, replace, _dist_training=True)
    global_nid_mapping = local_g.ndata[NID]
    src, dst = sampled_graph.edges()
    global_src, global_dst = F.gather_row(global_nid_mapping, src), \
            F.gather_row(global_nid_mapping, dst)
    global_eids = F.gather_row(local_g.edata[EID], sampled_graph.edata[EID])
    return global_src, global_dst, global_eids

def _find_edges(local_g, partition_book, seed_edges):
    """Given an edge ID array, return the source
        and destination node ID array ``s`` and ``d`` in the local partition.
    """
    local_eids = partition_book.eid2localeid(seed_edges, partition_book.partid)
    local_eids = F.astype(local_eids, local_g.idtype)
    local_src, local_dst = local_g.find_edges(local_eids)
    global_nid_mapping = local_g.ndata[NID]
    global_src = global_nid_mapping[local_src]
    global_dst = global_nid_mapping[local_dst]
    return global_src, global_dst

def _in_subgraph(local_g, partition_book, seed_nodes):
    """ Get in subgraph from local partition.

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
    return global_src, global_dst, global_eids


class SamplingRequest(Request):
    """Sampling Request"""

    def __init__(self, nodes, fan_out, edge_dir='in', prob=None, replace=False):
        self.seed_nodes = nodes
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.fan_out = fan_out

    def __setstate__(self, state):
        self.seed_nodes, self.edge_dir, self.prob, self.replace, self.fan_out = state

    def __getstate__(self):
        return self.seed_nodes, self.edge_dir, self.prob, self.replace, self.fan_out

    def process_request(self, server_state):
        local_g = server_state.graph
        partition_book = server_state.partition_book
        global_src, global_dst, global_eids = _sample_neighbors(local_g, partition_book,
                                                                self.seed_nodes,
                                                                self.fan_out, self.edge_dir,
                                                                self.prob, self.replace)
        return SubgraphResponse(global_src, global_dst, global_eids)

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
        global_src, global_dst = _find_edges(local_g, partition_book, self.edge_ids)

        return FindEdgeResponse(global_src, global_dst, self.order_id)

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
        global_src, global_dst, global_eids = _in_subgraph(local_g, partition_book,
                                                           self.seed_nodes)
        return SubgraphResponse(global_src, global_dst, global_eids)


def merge_graphs(res_list, num_nodes):
    """Merge request from multiple servers"""
    if len(res_list) > 1:
        srcs = []
        dsts = []
        eids = []
        for res in res_list:
            srcs.append(res.global_src)
            dsts.append(res.global_dst)
            eids.append(res.global_eids)
        src_tensor = F.cat(srcs, 0)
        dst_tensor = F.cat(dsts, 0)
        eid_tensor = F.cat(eids, 0)
    else:
        src_tensor = res_list[0].global_src
        dst_tensor = res_list[0].global_dst
        eid_tensor = res_list[0].global_eids
    g = graph((src_tensor, dst_tensor), num_nodes=num_nodes)
    g.edata[EID] = eid_tensor
    return g

LocalSampledGraph = namedtuple('LocalSampledGraph', 'global_src global_dst global_eids')

def _distributed_access(g, nodes, issue_remote_req, local_access):
    '''A routine that fetches local neighborhood of nodes from the distributed graph.

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

    Returns
    -------
    DGLHeteroGraph
        The subgraph that contains the neighborhoods of all input nodes.
    '''
    req_list = []
    partition_book = g.get_partition_book()
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
        src, dst, eids = local_access(g.local_partition, partition_book, local_nids)
        res_list.append(LocalSampledGraph(src, dst, eids))

    # receive responses from remote machines.
    if msgseq2pos is not None:
        results = recv_responses(msgseq2pos)
        res_list.extend(results)

    sampled_graph = merge_graphs(res_list, g.number_of_nodes())
    return sampled_graph

def sample_neighbors(g, nodes, fanout, edge_dir='in', prob=None, replace=False):
    """Sample from the neighbors of the given nodes from a distributed graph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The returned graph will contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    This version provides an experimental support for heterogeneous graphs.
    When the input graph is heterogeneous, the sampled subgraph is still stored in
    the homogeneous graph format. That is, all nodes and edges are assigned with
    unique IDs (in contrast, we typically use a type name and a node/edge ID to
    identify a node or an edge in ``DGLGraph``). We refer to this type of IDs
    as *homogeneous ID*.
    Users can use :func:`dgl.distributed.GraphPartitionBook.map_to_per_ntype`
    and :func:`dgl.distributed.GraphPartitionBook.map_to_per_etype`
    to identify their node/edge types and node/edge IDs of that type.

    For heterogeneous graphs, ``nodes`` can be a dictionary whose key is node type
    and the value is type-specific node IDs; ``nodes`` can also be a tensor of
    *homogeneous ID*.

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
    replace : bool, optional
        If True, sample with replacement.

        When sampling with replacement, the sampled subgraph could have parallel edges.

        For sampling without replacement, if fanout > the number of neighbors, all the
        neighbors are sampled. If fanout == -1, all neighbors are collected.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.  It is on CPU.
    """
    gpb = g.get_partition_book()
    if isinstance(nodes, dict):
        homo_nids = []
        for ntype in nodes:
            assert ntype in g.ntypes, 'The sampled node type does not exist in the input graph'
            if F.is_tensor(nodes[ntype]):
                typed_nodes = nodes[ntype]
            else:
                typed_nodes = toindex(nodes[ntype]).tousertensor()
            homo_nids.append(gpb.map_to_homo_nid(typed_nodes, ntype))
        nodes = F.cat(homo_nids, 0)
    def issue_remote_req(node_ids):
        return SamplingRequest(node_ids, fanout, edge_dir=edge_dir,
                               prob=prob, replace=replace)
    def local_access(local_g, partition_book, local_nids):
        return _sample_neighbors(local_g, partition_book, local_nids,
                                 fanout, edge_dir, prob, replace)
    return _distributed_access(g, nodes, issue_remote_req, local_access)

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
        mask = (partition_id == pid)
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
        src_ids = F.scatter_row(src_ids, reorder_idx[partition_book.partid], src)
        dst_ids = F.scatter_row(dst_ids, reorder_idx[partition_book.partid], dst)

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
    """ Given an edge ID array, return the source and destination
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
    def issue_remove_req(edge_ids, order_id):
        return EdgesRequest(edge_ids, order_id)
    def local_access(local_g, partition_book, edge_ids):
        return _find_edges(local_g, partition_book, edge_ids)
    return _distributed_edge_access(g, edge_ids, issue_remove_req, local_access)

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
        assert len(nodes) == 1, 'The distributed in_subgraph only supports one node type for now.'
        nodes = list(nodes.values())[0]
    def issue_remote_req(node_ids):
        return InSubgraphRequest(node_ids)
    def local_access(local_g, partition_book, local_nids):
        return _in_subgraph(local_g, partition_book, local_nids)
    return _distributed_access(g, nodes, issue_remote_req, local_access)

register_service(SAMPLING_SERVICE_ID, SamplingRequest, SubgraphResponse)
register_service(EDGES_SERVICE_ID, EdgesRequest, FindEdgeResponse)
register_service(INSUBGRAPH_SERVICE_ID, InSubgraphRequest, SubgraphResponse)
