from .rpc import Request, Response, remote_call
from .server_state import ServerState
from .dist_graph import DistGraph
from ..sampling import sample_neighbors as local_sample_neighbors
from . import register_service
from .graph_partition_book import GraphPartitionBook
from .. import to_block
import dgl
import torch as th

SAMPLING_SERVICE_ID = 6657


class SamplingResponse(Response):

    def __init__(self, global_src, global_dst, global_eids):
        self.global_src = global_src
        self.global_dst = global_dst
        self.global_eids = global_eids

    def __setstate__(self, state):
        self.global_src, self.global_dst, self.global_eids = state

    def __getstate__(self):
        return self.global_src, self.global_dst, self.global_eids


class SamplingRequest(Request):

    def __init__(self, partition_book: GraphPartitionBook, nodes, fan_out, edge_dir='in', prob=None, replace=False):
        self.seed_nodes = nodes
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.partition_book = partition_book
        self.fan_out = fan_out

    def __setstate__(self, state):
        

    def process_request(self, server_state: ServerState):
        
        local_g = server_state.graph
        partition_book = self.partition_book
        partition_id = partition_book.nid2partid()
        local_ids = partition_book.nid2localnid(self.seed_nodes, partition_id)
        sampled_graph = local_sample_neighbors(
            local_g, local_ids, self.fan_out, self.edge_dir, self.prob, self.replace)
        global_nid_mapping = self.g.ndata[NID]
        src, dst = sampled_graph.edges()
        global_src, global_dst = global_nid_mapping[src], global_nid_mapping[dst]
        global_eids = sampled_graph.edata[EID]
        res = SamplingResponse(global_src, global_dst, global_eids)
        return res

def merge_graphs(res_list):
    srcs = []
    dsts = []
    eids = []
    for res in res_list:
        res: SamplingResponse
        srcs.append(res.global_src)
        dsts.append(res.global_dst)
        eids.append(res.global_eids)
    src_tensor = th.concat(srcs)
    dst_tensor = th.concat(dsts)
    eid_tensor = th.concat(eids)
    g = dgl.graph((src_tensor, dst_tensor), restrict_format='coo')
    g.edata[EID] = eid_tensor
    return g
    

def sample_neighbors(g: DistGraph, nodes, fanout, edge_dir='in', prob=None, replace=False):
    assert fan_out == 1
    assert edge_dir == 'in'
    req_list = []
    blocks = []

    for node in nodes:
        req = SamplingRequest(
            g, [node], fanout, edge_dir=edge_dir, prob=prob, replace=replace)
        partition_book: GraphPartitionBook = g.get_partition_book()
        # Can be optimized, merge requests on the same machine
        partition_id = partition_book.nid2partid(node)
        req_list.append((partition_id, req))
    res_list = remote_call(req_list)
    return merge_graphs(res_list)



register_service(SAMPLING_SERVICE_ID, SamplingRequest, SamplingResponse)

