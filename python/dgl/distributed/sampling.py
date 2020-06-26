"""Sampling module"""
from collections import namedtuple

from .rpc import Request, Response, send_requests_to_machine, recv_responses
from ..sampling import sample_neighbors as local_sample_neighbors
from . import register_service
from ..convert import graph
from ..base import NID, EID
from ..utils import toindex
from .. import backend as F

__all__ = ['sample_neighbors']

SAMPLING_SERVICE_ID = 6657


class SamplingResponse(Response):
    """Sampling Response"""

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
        return SamplingResponse(global_src, global_dst, global_eids)


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

def sample_neighbors(dist_graph, nodes, fanout, edge_dir='in', prob=None, replace=False):
    """Sample from the neighbors of the given nodes from a distributed graph.

    When sampling with replacement, the sampled subgraph could have parallel edges.

    For sampling without replace, if fanout > the number of neighbors, all the
    neighbors are sampled.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
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
    assert edge_dir == 'in'
    req_list = []
    partition_book = dist_graph.get_partition_book()
    nodes = toindex(nodes).tousertensor()
    partition_id = partition_book.nid2partid(nodes)
    local_nids = None
    for pid in range(partition_book.num_partitions()):
        node_id = F.boolean_mask(nodes, partition_id == pid)
        # We optimize the sampling on a local partition if the server and the client
        # run on the same machine. With a good partitioning, most of the seed nodes
        # should reside in the local partition. If the server and the client
        # are not co-located, the client doesn't have a local partition.
        if pid == partition_book.partid and dist_graph.local_partition is not None:
            assert local_nids is None
            local_nids = node_id
        elif len(node_id) != 0:
            req = SamplingRequest(node_id, fanout, edge_dir=edge_dir,
                                  prob=prob, replace=replace)
            req_list.append((pid, req))

    # send requests to the remote machine.
    msgseq2pos = None
    if len(req_list) > 0:
        msgseq2pos = send_requests_to_machine(req_list)

    # sample neighbors for the nodes in the local partition.
    res_list = []
    if local_nids is not None:
        src, dst, eids = _sample_neighbors(dist_graph.local_partition, partition_book,
                                           local_nids, fanout, edge_dir, prob, replace)
        res_list.append(LocalSampledGraph(src, dst, eids))

    # receive responses from remote machines.
    if msgseq2pos is not None:
        results = recv_responses(msgseq2pos)
        res_list.extend(results)

    sampled_graph = merge_graphs(res_list, dist_graph.number_of_nodes())
    return sampled_graph


register_service(SAMPLING_SERVICE_ID, SamplingRequest, SamplingResponse)
