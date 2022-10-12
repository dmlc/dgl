# /*!
#  *   Copyright (c) 2022, NVIDIA Corporation
#  *   All rights reserved.
#  *
#  *   Licensed under the Apache License, Version 2.0 (the "License");
#  *   you may not use this file except in compliance with the License.
#  *   You may obtain a copy of the License at
#  *
#  *       http://www.apache.org/licenses/LICENSE-2.0
#  *
#  *   Unless required by applicable law or agreed to in writing, software
#  *   distributed under the License is distributed on an "AS IS" BASIS,
#  *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  *   See the License for the specific language governing permissions and
#  *   limitations under the License.
#  *
#  * \file train_dist_layer.py
#  * \brief distributed cooperative minibatching implementation
#  */

from datetime import timedelta

import nvtx

import torch as th
import torch.distributed as thd

from ...subgraph import in_subgraph
from ...base import NID, EID

from ...convert import graph, create_block
from ...heterograph_index import create_unitgraph_from_coo
from ...function import copy_src
from ...function import sum as fsum
from ...sparse import lighter_gspmm

from ...partition import metis_partition_assignment

from ... import backend as F
from ... import ndarray as nd

from ...dataloading.base import Sampler
from ...heterograph import DGLHeteroGraph as DGLGraph

def reorder_graph_wrapper(g, parts):
    new_g = g.reorder_graph(node_permute_algo='custom', edge_permute_algo='dst', store_ids=True, permute_config={'nodes_perm': th.cat(parts)})
    
    for k, v in g.ndata.items():
        if k != NID:
            new_g.ndata[k] = v[new_g.ndata[NID]]
    
    for k, v in g.edata.items():
        if k != EID:
            new_g.edata[k] = v[new_g.edata[EID]]
    
    return new_g

def uniform_partition(g, n_procs, random=True):
    N = g.num_nodes()
    idx = th.randperm(N, device=g.device) if random else th.arange(N, device=g.device)
    return [idx[i * N // n_procs: (i + 1) * N // n_procs] for i in range(n_procs)]

def metis_partition(g, n_procs):
    parts = metis_partition_assignment(g, n_procs)
    idx = th.argsort(parts)
    partition = th.searchsorted(parts[idx], th.arange(0, n_procs + 1))
    return [idx[partition[i]: partition[i + 1]] for i in range(n_procs)]

class DistConvFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, block, h):
        request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, distg_handle = ctx.cached_variables = block.cached_variables
        h = distg_handle.pull(h, request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids)
        return h
    
    @staticmethod
    def backward(ctx, grad_output):
        request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, distg_handle = ctx.cached_variables
        del ctx.cached_variables
        g_h = distg_handle.rpull(grad_output, request_counts, requested_sizes, seed_nodes, inv_ids)
        return None, g_h

class DistConv(th.nn.Module):
    def __init__(self, conv, pull=True):
        super().__init__()
        self.pull = pull
        self.layer = conv
    
    def forward(self, block, h):
        if self.pull:
            h = DistConvFunction.apply(block, h)
        return self.layer(block, h)

class DistSampler(Sampler):
    def __init__(self, g, sampler_t, fanouts, prefetch_node_feats=[], prefetch_labels=[], **kwargs):
        super().__init__()
        self.g = g
        self.samplers = [sampler_t([fanout], sort_src=True, output_device=self.g.device, **kwargs) for fanout in fanouts]
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_labels = prefetch_labels
        self.output_device = self.g.device
    
    def sample(self, g, seed_nodes, exclude_eids=None):
        # ignore g as we already store DistGraph
        return self.g.sample_blocks(seed_nodes, self.samplers, exclude_eids, self.prefetch_node_feats, self.prefetch_labels)

class LocalDGLGraph(DGLGraph):
    def __init__(self, g, num_nodes):
        super().__init__(g._graph, g._ntypes, g._etypes, g._node_frames, g._edge_frames)
        self._num_nodes = num_nodes

    def num_nodes(self, vtype=None):
        return (self._num_nodes[vtype] if vtype is not None else sum(c for _, c in self._num_nodes.items())) if len(self.ntypes) > 1 else self._num_nodes

class DistGraph(object):
    '''Distributed Graph object for GPUs
    
    We assume that torch.cuda.device() is called to set the GPU for the all processes
    We will rely on torch.cuda.current_device() to get the device.
    '''
    def __init__(self, g, g_parts, replication=0, compress=False):

        assert(thd.is_available()
            and thd.is_initialized()
            and thd.is_nccl_available())

        self.rank = thd.get_rank()
        self.world_size = thd.get_world_size()
        self.group_size = self.world_size if replication <= 0 else replication
        self.num_groups = self.world_size // self.group_size
        self.l_rank = self.rank % self.group_size
        self.group = self.rank // self.group_size
        self.group_start = self.group * self.group_size
        self.group_end = (self.group + 1) * self.group_size
        self.compress = compress

        assert(self.world_size % self.group_size == 0)
        assert(self.world_size == len(g_parts))
        
        self.device = th.cuda.current_device()
        cpu_device = th.device('cpu')
        
        assert(g.device == cpu_device)

        parts = [sum(g_parts[i * self.num_groups: (i + 1) * self.num_groups]) for i in range(self.group_size)]

        node_ranges = th.cumsum(th.tensor([0] + parts, device=cpu_device), dim=0)

        my_g = in_subgraph(g, th.arange(node_ranges[self.l_rank], node_ranges[self.l_rank + 1]))

        num_dst_nodes = node_ranges[self.l_rank + 1] - node_ranges[self.l_rank]

        if compress:
            max_num_dst_nodes = max(parts)
            self.log_pow_of_two = min(i for i in range(60) if 2 ** i >= max_num_dst_nodes)
            self.pow_of_two = 2 ** self.log_pow_of_two

            src, dst = my_g.edges()

            src_part = th.searchsorted(node_ranges, src + 1) - 1
            dst_part = self.l_rank # th.searchsorted(node_ranges, dst + 1) - 1

            new_src = src - node_ranges[src_part] + src_part * self.pow_of_two
            new_dst = dst - node_ranges[dst_part] + dst_part * self.pow_of_two
       
            g_dst_start = dst_part * self.pow_of_two
            # we make sure that all destination nodes assigned to us are in this list so that we are not missing any nodes
            unique_src = th.unique(th.cat((new_src, th.arange(g_dst_start, g_dst_start + num_dst_nodes))))

            uni_src = th.searchsorted(unique_src, new_src)
            uni_dst = th.searchsorted(unique_src, new_dst)
            # consider using dgl.create_block
            self.g = LocalDGLGraph(graph((uni_src, uni_dst), num_nodes=unique_src.shape[0], device=self.device), g.num_nodes())
            self.g.ndata[NID] = unique_src.to(self.device)
            self.g.edata[EID] = my_g.edata[EID].to(self.device)

            self.node_ranges = th.tensor([0] * (self.group * self.group_size) + list(range(0, self.group_size + 1)) + [self.group_size] * ((self.num_groups - self.group - 1) * self.group_size), device=self.device) * self.pow_of_two

            g_node_ranges = []
            cnts = [0] * self.group_size
            permute = []
            for i in range(self.num_groups):
                for j in range(self.group_size):
                    rank = j * self.num_groups + i
                    permute.append(rank)
                    g_node_ranges.append(self.pow_of_two * j + cnts[j])
                    cnts[j] += g_parts[rank]
            g_node_ranges.append(self.pow_of_two * self.group_size)
            inv_permute = sorted(range(len(permute)), key=permute.__getitem__)        
            self.g_node_ranges = th.tensor(g_node_ranges, device=self.device)[inv_permute + [-1]]
            self.permute = th.tensor(permute, device=self.device)
            self.inv_permute = th.tensor(inv_permute, device=self.device)

            self.pr = self.sorted_global_partition(self.g.ndata[NID], False)
            assert self.pr[self.rank + 1] - self.pr[self.rank] == num_dst_nodes
            self.g_pr = self.sorted_global_partition(self.g.ndata[NID], True)
            
            self.l_offset = self.g_pr[permute[self.rank]].item()

            g_offset = self.l_rank * self.pow_of_two

            self.dstdata = {NID: self.g.ndata[NID][self.g_pr[self.permute[self.rank]]: self.g_pr[self.permute[self.rank] + 1]]}
            g_NID = (self.dstdata[NID] - g_offset + node_ranges[self.l_rank]).to(cpu_device)
        else:
            self.g = my_g.to(self.device)

            self.node_ranges = th.tensor([0] * (self.group * self.group_size) + node_ranges.tolist() + [node_ranges[-1].item()] * ((self.num_groups - self.group - 1) * self.group_size), device=self.device)

            g_node_ranges = []
            cnts = [0] * self.group_size
            permute = []
            for i in range(self.num_groups):
                for j in range(self.group_size):
                    rank = j * self.num_groups + i
                    permute.append(rank)
                    g_node_ranges.append(node_ranges[j].item() + cnts[j])
                    cnts[j] += g_parts[rank]
            g_node_ranges.append(node_ranges[-1].item())
            inv_permute = sorted(range(len(permute)), key=permute.__getitem__)        
            self.g_node_ranges = th.tensor(g_node_ranges, device=self.device)[inv_permute + [-1]]
            self.permute = th.tensor(permute, device=self.device)
            self.inv_permute = th.tensor(inv_permute, device=self.device)

            self.pr = self.node_ranges
            assert self.pr[self.rank + 1] - self.pr[self.rank] == num_dst_nodes
            self.g_pr = self.g_node_ranges
            
            self.l_offset = self.g_pr[permute[self.rank]].item()

            self.dstdata = {}
            g_NID = th.arange(self.g_pr[self.permute[self.rank]], self.g_pr[self.permute[self.rank] + 1], device=cpu_device)

        g_EID = self.g.edata[EID].to(cpu_device)
        
        for k, v in list(g.ndata.items()):
            if k != NID:
                self.dstdata[k] = v[g_NID].to(self.device)
                g.ndata.pop(k)
        
        for k, v in list(g.edata.items()):
            if k != EID:
                self.g.edata[k] = v[g_EID].to(self.device)
                g.edata.pop(k)
        
        print(self.rank, self.g.num_nodes(), self.pr, self.g_pr, self.l_offset, self.node_ranges, self.g_node_ranges, self.permute, self.inv_permute)
        pg_options = th._C._distributed_c10d.ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
        pg_options._timeout = timedelta(minutes=1)
        self.comm = thd.new_group(ranks=None, backend='nccl', pg_options=pg_options)
        self.l_comm = self.comm
        if self.group_size < self.world_size:
            self.l_comms = [thd.new_group(ranks=range(group * self.group_size, (group + 1) * self.group_size), backend='nccl', pg_options=pg_options) for group in range(self.num_groups)]
            self.l_comm = self.l_comms[self.group]
        self.random_seed = th.randint(0, 10000000000000, (1,), device=self.device)
        thd.all_reduce(self.random_seed, thd.ReduceOp.SUM, self.comm)
        self.last_comm = self.comm
        self.works = []

    def sorted_global_partition(self, ids, g_comm):
        return th.searchsorted(ids, self.g_node_ranges if g_comm else self.node_ranges)
    
    def global_part(self, ids):
        return th.bitwise_right_shift(ids, self.log_pow_of_two) if self.compress else self.local_part(ids)
    
    def local_part(self, ids):
        return th.searchsorted(self.pr, ids + 1) - 1
    
    def global_to_local(self, ids, i=None):
        if not self.compress:
            return ids
        if i is None:
            i = self.global_part(ids)
        return ids - (i % self.group_size) * self.pow_of_two + self.pr[i]
    
    def local_to_global(self, ids, i=None):
        if not self.compress:
            return ids
        if i is None:
            i = self.local_part(ids)
        return ids - self.pr[i] + i * self.pow_of_two
    
    def synchronize_works(self, ins=None):
        for work in self.works:
            if work is not None:
                work.wait()
        self.works = []
        if ins is not None:
            th.flatten(ins[0])[0] += 0
    
    def all_to_all(self, outs, ins, async_op=False):
        g_comm = any(th.numel(t) > 0 and not (self.group_start <= i and i < self.group_end) for ts in [ins, outs] for i, t in enumerate(ts))
        comm = self.comm if g_comm else self.l_comm
        if not g_comm:
            outs = outs[self.group_start: self.group_end]
            ins = ins[self.group_start: self.group_end]
        if self.last_comm != comm:
            self.last_comm = comm
            self.synchronize_works(ins)
        work = thd.all_to_all(outs, ins, comm, async_op)
        if self.comm != self.l_comm:
            self.works.append(work)
        return work
    
    # local ids in, local ids out
    @nvtx.annotate("id exchange", color="purple")
    def exchange_node_ids(self, nodes, g_comm):
        nodes = self.g.ndata[NID][nodes] if self.compress else nodes
        partition = self.sorted_global_partition(nodes, g_comm)
        request_counts = th.diff(partition)
        received_request_counts = th.empty_like(request_counts)
        self.all_to_all(list(th.split(received_request_counts, 1)), list(th.split(request_counts[self.permute] if g_comm else request_counts, 1)))
        requested_sizes = received_request_counts.tolist()
        requested_nodes = th.empty(sum(requested_sizes), dtype=nodes.dtype, device=self.device)
        request_counts = request_counts.tolist()
        par_nodes = list(th.split(nodes, request_counts))
        if g_comm:
            par_nodes = [par_nodes[i] for i in self.permute.tolist()]
        self.all_to_all(list(th.split(requested_nodes, requested_sizes)), par_nodes)
        requested_nodes = self.global_to_local(requested_nodes, self.rank)
        return requested_nodes, requested_sizes, request_counts

    def get_random_seed(self, inc=1):
        self.random_seed += inc
        return self.random_seed.item()
    
    @nvtx.annotate("pull", color="purple")
    def pull(self, dsttensor, request_counts, requested_nodes, requested_sizes, dstnodes, inv_ids):
        out = th.empty((sum(request_counts),) + dsttensor.shape[1:], dtype=dsttensor.dtype, device=dsttensor.device)
        self.all_to_all(list(th.split(out, request_counts)), list(th.split(dsttensor[inv_ids], requested_sizes)))
        return out
    
    @nvtx.annotate("pull", color="purple")
    def pull_ex(self, dsttensor, srcnodes=None):
        if srcnodes is None:
            srcnodes = th.arange(self.g.num_nodes(), device=self.device)
        
        requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(srcnodes)
        dstnodes, inv_ids = th.unique(requested_nodes, return_inverse=True)
        
        return self.pull(dsttensor, request_counts, requested_nodes, requested_sizes, dstnodes, inv_ids)
    
    @nvtx.annotate("rpull", color="purple")
    def rpull(self, srctensor, request_counts, requested_sizes, dstnodes, inv_ids):
        out = th.empty((sum(requested_sizes),) + srctensor.shape[1:], dtype=srctensor.dtype, device=srctensor.device)
        src = th.arange(out.shape[0], device=self.device)
        dst = inv_ids
        self.all_to_all(list(th.split(out, requested_sizes)), list(th.split(srctensor, request_counts)))
        _graph = create_unitgraph_from_coo(2, out.shape[0], dstnodes.shape[0], src, dst, ['coo'], row_sorted=True)
        rout = th.zeros((dstnodes.shape[0],) + srctensor.shape[1:], dtype=srctensor.dtype, device=srctensor.device)
        lighter_gspmm(_graph, 'copy_lhs', 'sum',
                            F.zerocopy_to_dgl_ndarray(out),
                            nd.NULL['int64'],
                            F.zerocopy_to_dgl_ndarray_for_write(rout),
                            nd.NULL['int64'],
                            nd.NULL['int64'])
        return rout

    @nvtx.annotate("rpull", color="purple")
    def rpull_ex(self, srctensor, srcnodes=None):
        if srcnodes is None:
            srcnodes = th.arange(self.g.num_nodes(), device=self.device)
        
        requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(srcnodes)
        dstnodes, inv_ids = th.unique(requested_nodes, return_inverse=True)

        return self.rpull(srctensor, request_counts, requested_sizes, dstnodes, inv_ids)

    @nvtx.annotate("sample_blocks", color="purple")
    def sample_blocks(self, seed_nodes, samplers, exclude_eids=None, prefetch_node_feats=[], prefetch_labels=[]):
        blocks = []
        random_seed = self.get_random_seed(len(samplers))
        if not (th.all(self.pr[self.rank] <= seed_nodes) and th.all(seed_nodes < self.pr[self.rank + 1])):
            seed_nodes = th.unique(self.exchange_node_ids(th.sort(seed_nodes)[0], False)[0])
        output_nodes = seed_nodes
        for i, sampler in enumerate(reversed(samplers)):
            assert th.all(self.pr[self.rank] <= seed_nodes) and th.all(seed_nodes < self.pr[self.rank + 1])
            if hasattr(sampler, 'set_seed'):
                sampler.set_seed(random_seed + (0 if sampler.layer_dependency else i))
            seed_nodes, _, blocks_i = sampler.sample_blocks(self.g, seed_nodes, exclude_eids=exclude_eids)
            
            requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(seed_nodes, i == len(samplers) - 1)
            seed_nodes, inv_ids = th.unique(requested_nodes, return_inverse=True)
            
            blocks_i[0].cached_variables = request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, self

            blocks.insert(0, blocks_i[0])
        
        def feature_slicer(block):
            srcdataevents = {}
            for k in prefetch_node_feats:
                tensor = self.dstdata[k][requested_nodes - self.l_offset].to(self.device)
                out = th.empty((sum(request_counts),) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                par_out = list(th.split(out, request_counts))
                par_out = [par_out[i] for i in self.permute.tolist()]
                work = self.all_to_all(par_out, list(th.split(tensor, requested_sizes)), True)
                block.srcdata[k] = out
                srcdataevents[k] = work
            
            def wait(k=None):
                if k is None:
                    for k, work in srcdataevents.items():
                        work.wait()
                else:
                    srcdataevents[k].wait()
            return wait
        
        def label_slicer(block):
            for k in prefetch_labels:
                block.dstdata[k] = self.dstdata[k][output_nodes - self.l_offset].to(self.device)

        blocks[0].slice_features = feature_slicer
        blocks[-1].slice_labels = label_slicer

        return seed_nodes, output_nodes, blocks
