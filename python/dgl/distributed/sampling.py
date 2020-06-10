from .rpc import Request, Response, remote_call, send_request, recv_response
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

    def process_request(self, server_state: ServerState):        
        local_g = server_state.graph
        partition_book = server_state.partition_book
        print(f"Server: {server_state.rank}::{self.seed_nodes}")
        partition_id = partition_book.nid2partid(th.tensor(self.seed_nodes))
        print(f"Partition_id {partition_book._part_id} {server_state.rank} {partition_id} {self.seed_nodes}")
        local_ids = partition_book.nid2localnid(th.tensor(self.seed_nodes), partition_id)
        sampled_graph = local_sample_neighbors(
            local_g, local_ids, self.fan_out, self.edge_dir, self.prob, self.replace)
        global_nid_mapping = local_g.ndata[dgl.NID]
        src, dst = sampled_graph.edges()
        global_src, global_dst = global_nid_mapping[src], global_nid_mapping[dst]
        global_eids = sampled_graph.edata[dgl.EID]
        res = SamplingResponse(global_src, global_dst, global_eids)
        print(f"DONE {server_state.rank}")
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
    src_tensor = th.cat(srcs)
    dst_tensor = th.cat(dsts)
    eid_tensor = th.cat(eids)
    g = dgl.graph((src_tensor, dst_tensor), restrict_format='coo')
    g.edata[dgl.EID] = eid_tensor
    return g
    

def sample_neighbors(g: DistGraph, nodes, fanout, edge_dir='in', prob=None, replace=False):
    assert fanout == 1
    assert edge_dir == 'in'
    req_list = []
    blocks = []

    for node in nodes:
        partition_book: GraphPartitionBook = g.get_partition_book()
        req = SamplingRequest(
            [node], fanout, edge_dir=edge_dir, prob=prob, replace=replace)

        # Can be optimized, merge requests on the same machine
        partition_id = partition_book.nid2partid(th.tensor(node))
        print(f"{partition_id}::{node}")
        send_request(partition_id, req)
        # req_list.append((partition_id, req))
    print("before")
    res_list = [recv_response() for _ in nodes]
    print("after")
    print(res_list)
    return merge_graphs(res_list)



register_service(SAMPLING_SERVICE_ID, SamplingRequest, SamplingResponse)

