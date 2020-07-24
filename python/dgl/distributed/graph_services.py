"""A set of graph services of getting subgraphs from DistGraph"""
from collections import namedtuple

from .rpc import Request, Response, send_requests_to_machine, recv_responses
from ..sampling import sample_neighbors as local_sample_neighbors
from ..transform import in_subgraph as local_in_subgraph
from . import register_service
from ..convert import graph
from ..base import NID, EID
from ..utils import toindex
from .. import backend as F

__all__ = ['sample_neighbors', 'in_subgraph']

SAMPLING_SERVICE_ID = 6657
INSUBGRAPH_SERVICE_ID = 6658

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


def _sample_neighbors(local_g, partition_book, seed_nodes, fan_out, edge_dir, prob, replace):
    """ Sample from local partition.

    The input nodes use global Ids. We need to map the global node Ids to local node Ids,
    perform sampling and map the sampled results to the global Ids space again.
    The sampled results are stored in three vectors that store source nodes, destination nodes
    and edge Ids.
    """
    local_ids = partition_book.nid2localnid(seed_nodes, partition_book.partid)
    local_ids = F.astype(local_ids, local_g.idtype)
    # local_ids = self.seed_nodes
    sampled_graph = local_sample_neighbors(
        local_g, local_ids, fan_out, edge_dir, prob, replace)
    global_nid_mapping = local_g.ndata[NID]
    src, dst = sampled_graph.edges()
    global_src, global_dst = global_nid_mapping[src], global_nid_mapping[dst]
    global_eids = F.gather_row(local_g.edata[EID], sampled_graph.edata[EID])
    return global_src, global_dst, global_eids


def _in_subgraph(local_g, partition_book, seed_nodes):
    """ Get in subgraph from local partition.

    The input nodes use global Ids. We need to map the global node Ids to local node Ids,
    get in-subgraph and map the sampled results to the global Ids space again.
    The results are stored in three vectors that store source nodes, destination nodes
    and edge Ids.
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
    g = graph((src_tensor, dst_tensor),
              restrict_format='coo', num_nodes=num_nodes)
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

    When sampling with replacement, the sampled subgraph could have parallel edges.

    For sampling without replace, if fanout > the number of neighbors, all the
    neighbors are sampled.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    For now, we only support the input graph with one node type and one edge type.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    nodes : tensor or dict
        Node ids to sample neighbors from.
    fanout : int
        The number of sampled neighbors for each node.
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
    if isinstance(nodes, dict):
        assert len(nodes) == 1, 'The distributed sampler only supports one node type for now.'
        nodes = list(nodes.values())[0]
    def issue_remote_req(node_ids):
        return SamplingRequest(node_ids, fanout, edge_dir=edge_dir,
                               prob=prob, replace=replace)
    def local_access(local_g, partition_book, local_nids):
        return _sample_neighbors(local_g, partition_book, local_nids,
                                 fanout, edge_dir, prob, replace)
    return _distributed_access(g, nodes, issue_remote_req, local_access)

def in_subgraph(g, nodes):
    """Extract the subgraph containing only the in edges of the given nodes.

    The subgraph keeps the same type schema and the cardinality of the original one.
    Node/edge features are not preserved. The original IDs
    the extracted edges are stored as the `dgl.EID` feature in the returned graph.

    For now, we only support the input graph with one node type and one edge type.

    Parameters
    ----------
    g : DistGraph
        The distributed graph structure.
    nodes : tensor
        Node ids to sample neighbors from.

    Returns
    -------
    DGLHeteroGraph
        The subgraph.
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
register_service(INSUBGRAPH_SERVICE_ID, InSubgraphRequest, SubgraphResponse)
