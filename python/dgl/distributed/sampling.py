"""Sampling module"""
from .rpc import Request, Response, remote_call_to_machine
from ..sampling import sample_neighbors as local_sample_neighbors
from . import register_service
from ..convert import graph
from ..base import NID, EID
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
        local_ids = F.astype(partition_book.nid2localnid(
            F.tensor(self.seed_nodes), partition_book.partid), local_g.idtype)
        # local_ids = self.seed_nodes
        sampled_graph = local_sample_neighbors(
            local_g, local_ids, self.fan_out, self.edge_dir, self.prob, self.replace)
        global_nid_mapping = local_g.ndata[NID]
        src, dst = sampled_graph.edges()
        global_src, global_dst = global_nid_mapping[src], global_nid_mapping[dst]
        global_eids = F.gather_row(
            local_g.edata[EID], sampled_graph.edata[EID])

        res = SamplingResponse(global_src, global_dst, global_eids)
        return res


def merge_graphs(res_list, num_nodes):
    """Merge request from multiple servers"""
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
    g = graph((src_tensor, dst_tensor),
              restrict_format='coo', num_nodes=num_nodes)
    g.edata[EID] = eid_tensor
    return g


def sample_neighbors(dist_graph, nodes, fanout, edge_dir='in', prob=None, replace=False):
    """Sample neighbors"""
    assert edge_dir == 'in'
    req_list = []
    partition_book = dist_graph.get_partition_book()

    partition_id = F.asnumpy(
        partition_book.nid2partid(F.tensor(nodes))).tolist()
    node_id_per_partition = [[]
                             for _ in range(partition_book.num_partitions())]
    for pid, node in zip(partition_id, nodes):
        node_id_per_partition[pid].append(node)
    for pid, node_id in enumerate(node_id_per_partition):
        if len(node_id) != 0:
            req = SamplingRequest(
                node_id, fanout, edge_dir=edge_dir, prob=prob, replace=replace)
            req_list.append((pid, req))
    res_list = remote_call_to_machine(req_list)
    sampled_graph = merge_graphs(res_list, dist_graph.number_of_nodes())
    return sampled_graph


register_service(SAMPLING_SERVICE_ID, SamplingRequest, SamplingResponse)
