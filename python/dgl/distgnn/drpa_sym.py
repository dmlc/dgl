import torch
import time
from torch import nn
from torch.nn import functional as F
import os, psutil
import gc
import sys

from .. import function as fn
from ..utils import expand_as_pair, check_eq_shape
from math import ceil, floor
from .. import DGLHeteroGraph

from ..sparse import scatter_reduce_lr, fdrpa_gather_emb_lr
from ..sparse import scatter_reduce_rl, fdrpa_gather_emb_rl
from ..sparse import fdrpa_comm_buckets, deg_div, deg_div_back, fdrpa_init_buckets

display = False
class GQueue():
    def __init__(self):
        self.queue = []
        
    def push(self, val):
        self.queue.append(val)

    def pop(self):
        if self.empty():
            return -1
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)
    
    def printq(self):
        print(self.queue)

    def empty(self):
        if (len(self.queue)) == 0:
            return True
        else:
            return False

    def purge(self):
        self.queue = []

        
## communication storage
gfqueue_feats_lr  = GQueue()
gfqueue_nodes_lr = GQueue()
gfqueue_feats_rl = GQueue()

## buffer storage
buffcomm_feats_lr  = GQueue()
buffcomm_nodes_lr  = GQueue()
buffcomm_feats_size_lr  = GQueue()
buffcomm_nodes_size_lr  = GQueue()
buffcomm_snl_lr  = GQueue()
buffcomm_snl_size_lr  = GQueue()

buffcomm_feats_rl  = GQueue()
buffcomm_feats_size_rl  = GQueue()

buffcomm_iter_lr = GQueue()
buffcomm_iter_rl = GQueue()


## DRPA
def drpa(gobj, rank, num_parts, node_map, nrounds, dist, nlayers):
    d = DRPAMaster(gobj._graph, gobj._ntypes, gobj._etypes, gobj._node_frames, gobj._edge_frames)
    d.drpa_init(rank, num_parts, node_map, nrounds, dist, nlayers)
    return d


class DRPAMaster(DGLHeteroGraph):            
    def drpa_init(self, rank, num_parts, node_map, nrounds, dist, nlayers):
        self.rank = rank
        self.num_parts = num_parts
        self.node_map = node_map
        self.nrounds = nrounds
        self.dist = dist
        self.nlayers = nlayers + 1

        self.epochs_ar = [0 for i in range(self.nlayers)]
        self.epochi = 0
        self.gather_q41 = GQueue()
        self.output_sr_ar = []

        if self.nrounds == -1: return
        ## Creates buckets based on ndrounds        

        adj = self.dstdata['adj']
        lf = self.dstdata['lf']
        width = adj.shape[1]

        self.drpa_create_buckets()
        self.drpa_init_buckets(adj, lf, width)

        
    def drpa_finalize(self):
        if self.rank == 0:
            print("Symm: Clearning backlogs", flush=True)
        while gfqueue_feats_lr.empty() == False:
            req = gfqueue_feats_lr.pop()
            req.wait()
            req = gfqueue_nodes_lr.pop()
            req.wait()
        while gfqueue_feats_rl.empty() == False:                
            req = gfqueue_feats_rl.pop()
            req.wait()

        buffcomm_feats_lr.purge()
        buffcomm_nodes_lr.purge()
        buffcomm_feats_size_lr.purge()
        buffcomm_nodes_size_lr.purge()
        buffcomm_snl_lr.purge()
        buffcomm_snl_size_lr.purge()
        buffcomm_iter_lr.purge()
        buffcomm_feats_rl.purge()
        buffcomm_feats_size_rl.purge()
        buffcomm_iter_rl.purge()
        

        if gfqueue_feats_lr.empty() == False:
            print("gfqueue_feats_lr not empty after backlogs flushing", flush=True)
        if gfqueue_nodes_lr.empty() == False:
            print("gfqueue_nodes_lr not empty after backlogs flushing", flush=True)
        if gfqueue_feats_rl.empty() == False:
            print("gfqueue_feats_rl not empty after backlogs flushing", flush=True)
            
        if self.rank == 0:
            print("Clearning backlogs ends ", flush=True)

            
    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        assert self.rank != -1, "drpa not initialized !!!"

        mean = 0
        if reduce_func.name == "mean":
            reduce_func = fn.sum('m', 'neigh')
            mean = 1
            
        
        tic = time.time()
        DGLHeteroGraph.update_all(self, message_func, reduce_func)
        toc = time.time()

        if self.rank == 0  and display:
            print("Time for local aggregate: {:0.4f}, nrounds {}".format(toc - tic, self.nrounds))
        
        if self.nrounds == -1:
            if mean == 1:
                feat_dst = self.dstdata['h']
                self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
                self.dstdata['neigh'] = self.dstdata['neigh'] / self.r_in_degs.unsqueeze(-1)            
            return
        
        neigh = self.dstdata['neigh']
        adj = self.dstdata['adj']
        inner_node = self.dstdata['inner_node']
        lftensor = self.dstdata['lf']
        feat_dst = self.dstdata['h']
        
        epoch = self.epochs_ar[self.epochi]

        tic = time.time()
        
        self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
        self.dstdata['neigh'] = call_drpa_core(neigh, adj, inner_node,
                                               lftensor, self.selected_nodes,
                                               self.node_map, self.num_parts,
                                               self.rank, epoch,
                                               self.dist,
                                               self.r_in_degs,
                                               self.nrounds,
                                               self.output_sr_ar,
                                               self.gather_q41)
        
        self.epochs_ar[self.epochi] += 1
        self.epochi = (self.epochi + 1) % (self.nlayers)

        
        toc = time.time()
        if self.rank == 0 and display:        
            print("Time for remote aggregate: {:0.4f}".format(toc - tic))

        if mean == 1:
            self.dstdata['neigh'] = self.dstdata['neigh'] / self.r_in_degs.unsqueeze(-1)
            

    def drpa_init_buckets(self, adj, lf, width):

        for l in range(self.nrounds):
            buckets = torch.tensor([0 for i in range(self.num_parts)], dtype=torch.int32)
            sn = torch.tensor(self.selected_nodes[l], dtype=torch.int32)
            nm = torch.tensor(self.node_map, dtype=torch.int32)
            fdrpa_init_buckets(adj, sn, nm, buckets,
                               lf, width, self.num_parts, self.rank)
            input_sr = []
            for i in range(0, self.num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))
                
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, self.num_parts)]
            sync_req = self.dist.all_to_all(output_sr, input_sr, async_op=True)  
            sync_req.wait() ## recv the #nodes communicated
            self.output_sr_ar.append(output_sr)  ## output


    def drpa_create_buckets(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)
        
        #nrounds_ = self.nrounds
        self.selected_nodes = [ [] for i in range(self.nrounds)]  ## #native nodes

        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = inner_nodex.count(0)  ## count split nodes
        alien_nodes_per_round = int((total_alien_nodes + self.nrounds -1) / self.nrounds)
        
        counter = 0
        pos = 0
        r = 0
        while counter < n:
            if inner_nodex[counter] == 0:    ##split node
                self.selected_nodes[r].append(counter)
                pos += 1
                if pos % alien_nodes_per_round == 0:
                    r = r + 1
                    
            counter += 1
                    

        if (counter != len(inner_nodex)):
            print("counter: ", counter, " ", len(inner_nodex))
            
        assert counter == len(inner_nodex), "assertion"
        #if pos == total_alien_nodes:
        #    print("pos: ", pos, " ", total_alien_nodes)
        assert pos == total_alien_nodes, "pos alien not matching!!"
        
        #if self.rank == 0:
        #    print("Selected nodes in each round: ", flush=True)
        #    for i in range(self.nrounds):
        #        print("round: ", i,  " nodes: ", len(self.selected_nodes[i]), flush=True);

                
    def in_degrees(self):
        try:
            return self.r_in_degs
        except:
            #print("Passing only local node degrees.")
            pass
    
        return DGLHeteroGraph.in_degrees(self)


def message(rank, msg, val=-1, val_=-1):    
    if rank == 0 and display:
        if val == -1:
            print(msg, flush=True)
        elif val_ == -1:
            print(msg.format(val), flush=True)
        else:
            print(msg.format(val, val_), flush=True)

            
class DRPACore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, adj, inner_node, lftensor, selected_nodes,
                node_map, num_parts, rank, epoch, dist, in_degs, nrounds,
                output_sr_ar, gather_q41):        

        prof = []        ## runtime profile
        nrounds_update = nrounds

        feat_size = feat.shape[1]
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        base_chunk_size = int(int_threshold / num_parts) ##bytes            
        base_chunk_size_fs = floor(base_chunk_size / (feat_size + 1) )
        roundn =  epoch % nrounds
        
        node_map_t = torch.tensor(node_map, dtype=torch.int32)
        selected_nodes_t = []
        for sn in selected_nodes:
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))            
            
        ## section I: prepare the msg
        buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        width = adj.shape[1]   ## feature vector length

        tic = time.time()
        #### 1. get bucket sizes
        ver2part = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        ver2part_index = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        fdrpa_comm_buckets(adj, selected_nodes_t[roundn], ver2part, ver2part_index,
                           node_map_t, buckets, lftensor, width, num_parts, rank)
        
        message(rank, "Time for bucketing: {:0.4f}", (time.time() - tic))        
            
        ###### comms to gather the bucket sizes for all-to-all feats
        input_sr = []
        for i in range(0, num_parts):
            input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))

        output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
        sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
        sync_req.wait() ## recv the #nodes communicated
            
        ### 3. gather emdeddings
        send_feat_len = 0
        in_size = []
        for i in range(num_parts):
            in_size.append(int(buckets[i]) * (feat_size + 1))
            send_feat_len += in_size[i]
                     
        ## mpi call split starts
        ##############################################################################
        tic = time.time()
            
        cum = 0; flg = 0
        for i in output_sr:
            cum += int(i) * (feat_size + 1)
            if int(i) >= base_chunk_size_fs: flg = 1

        for i in input_sr:
            if int(i) >= base_chunk_size_fs: flg = 1
        
        nsplit_comm = 1
        if cum >= int_threshold or send_feat_len >= int_threshold or flg:        
            for i in range(num_parts):
                val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                if val > nsplit_comm: nsplit_comm = val
                    
        nsplit_comm_t = torch.tensor(nsplit_comm)
        req_nsplit_comm = torch.distributed.all_reduce(nsplit_comm_t,
                                                       op=torch.distributed.ReduceOp.MAX,
                                                       async_op=True)
        
        lim = 1            
        soffset_base = [0 for i in range(num_parts)]  ## min chunk size        
        soffset_cur = [0 for i in range(num_parts)]       ## send list of ints
        roffset_cur = [0 for i in range(num_parts)]       ## recv list of intrs
        
        j=0
        while j < lim:
            tsend = 0; trecv = 0
            for i in range(num_parts):
                soffset_base[i] += soffset_cur[i]
                if input_sr[i] < base_chunk_size_fs:
                    soffset_cur[i] = int(input_sr[i])
                    input_sr[i] = 0
                else:
                    soffset_cur[i] = base_chunk_size_fs
                    input_sr[i] -= base_chunk_size_fs
                    
                if output_sr[i]  < base_chunk_size_fs:
                    roffset_cur[i] = int(output_sr[i])
                    output_sr[i] = 0
                else:
                    roffset_cur[i] = base_chunk_size_fs
                    output_sr[i] -= base_chunk_size_fs
    	
                tsend += soffset_cur[i]
                trecv += roffset_cur[i]
                
            send_node_list = \
                [torch.empty(soffset_cur[i], dtype=torch.int32) for i in range(num_parts)]
            sten_ = torch.empty(tsend * (feat_size + 1), dtype=feat.dtype)
            dten_ = torch.empty(trecv * (feat_size + 1), dtype=feat.dtype)
            sten_nodes = torch.empty(tsend , dtype=torch.int32)
            dten_nodes = torch.empty(trecv , dtype=torch.int32)
            
            out_size = [0 for i in range(num_parts)]
            in_size = [0 for i in range(num_parts)]
            out_size_nodes = [0 for i in range(num_parts)]
            in_size_nodes = [0 for i in range(num_parts)]

            offset = 0            
            for i in range(num_parts): ## gather by followers                
                fdrpa_gather_emb_lr(feat, feat.shape[0], adj, sten_, offset,
                                    send_node_list[i], sten_nodes,
                                    selected_nodes_t[roundn],
                                    in_degs, ver2part, ver2part_index, width, feat_size, i,
                                    soffset_base[i], soffset_cur[i], node_map_t, num_parts)
                
                out_size[i]       = roffset_cur[i] * (feat_size + 1)
                in_size[i]        = soffset_cur[i] * (feat_size + 1)
                offset            += soffset_cur[i]
                out_size_nodes[i] = roffset_cur[i]
                in_size_nodes[i]  = soffset_cur[i]

            message(rank, "Sending {}, recving {} data I", tsend, trecv)
            
            req_feats = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)    	
            gfqueue_feats_lr.push(req_feats)
            req_nodes = dist.all_to_all_single(dten_nodes, sten_nodes,
                                         out_size_nodes, in_size_nodes,
                                         async_op=True)    	
            gfqueue_nodes_lr.push(req_nodes)
                
            soffset_cur_copy = soffset_cur.copy()

    	    ## section III: store pointers for the data in motion
            buffcomm_feats_lr.push(dten_)
            buffcomm_nodes_lr.push(dten_nodes)
            buffcomm_feats_size_lr.push(out_size)
            buffcomm_nodes_size_lr.push(out_size_nodes)
            buffcomm_snl_lr.push(send_node_list)    ## fwd phase II
            buffcomm_snl_size_lr.push(soffset_cur_copy) ## fwd phase II
            
            if j == 0:
                req_nsplit_comm.wait()
                lim = int(nsplit_comm_t)
            j += 1
        ##############################################################################        
        ## mpi call split ends
        buffcomm_iter_lr.push(lim)

        message(rank, "Max iters in MPI split comm: {}", (lim))                    
        #message(rank, "Time for Gather I: {:0.4f}", (time.time() - tic))
        prof.append('Gather I: {:0.4f}'.format(time.time() - tic))            
        
        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []        
        if epoch >= nrounds_update or nrounds == 1:
            assert gfqueue_feats_lr.empty() == False, "Error: Forward empty queue !!!"

            ticg = time.time()                
            lim = buffcomm_iter_lr.pop()
            out_size_nodes_ar = []                
            for i in range(lim):
                
                if rank == 0 and display: tic = time.time()
                req = gfqueue_feats_lr.pop();  req.wait()
                req = gfqueue_nodes_lr.pop(); req.wait()
            
                #message(rank, "Time for async comms I: {:4f}", (time.time() - tic))
                prof.append('Async comm I: {:0.4f}'.format(time.time() - tic))
                    
                otf = buffcomm_feats_lr.pop()
                out_size = buffcomm_feats_size_lr.pop()
                otn = buffcomm_nodes_lr.pop()
                out_size_nodes = buffcomm_nodes_size_lr.pop()
                out_size_nodes_ar.append(out_size_nodes)
                
                recv_list_nodes_ar = []; ilen = 0
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))
            
                pos = torch.tensor([0], dtype=torch.int64)        
                offsetf = 0; offsetn = 0
                for l in range(num_parts):
                    scatter_reduce_lr(otf, offsetf, otn, offsetn,
                                      feat, in_degs, node_map_t, out_size[l], feat_size,
                                      num_parts, recv_list_nodes_ar[l], pos,
                                      int(out_size_nodes[l]), rank)
                    
                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]
                    
                assert ilen == pos[0], "Issue in scatter reduce!"                
                recv_list_nodes.append(recv_list_nodes_ar)


            #message(rank, "Time for scatter I: {:0.4f} in epoch: {}", (time.time() - ticg), epoch)
            prof.append('Scatter I: {:0.4f}'.format(time.time() - tic))

            tic = time.time()
            for j in range(lim):               ### gather-scatter round II
                tsend = 0; trecv = 0
                stn_fp2 = buffcomm_snl_size_lr.pop()
                out_size_nodes = out_size_nodes_ar[j]
                
                for i in range(num_parts):
                    tsend += out_size_nodes[i]
                    trecv += stn_fp2[i]
                    
                recv_list_nodes_ = recv_list_nodes[j]                    
                sten_ = torch.empty(tsend * (feat_size + 1), dtype=feat.dtype)
                dten_ = torch.empty(trecv * (feat_size + 1), dtype=feat.dtype)            
                
                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                offset = 0
                for i in range(num_parts):
                    fdrpa_gather_emb_rl(feat, feat.shape[0], sten_, offset,
                                        recv_list_nodes_[i], out_size_nodes[i],
                                        in_degs,
                                        feat_size, i,
                                        node_map_t, num_parts)
                    
                    out_size[i]       = stn_fp2[i] * (feat_size + 1)
                    in_size[i]        = out_size_nodes[i] * (feat_size + 1)
                    offset           += in_size[i]
                                    
                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)    	
                gfqueue_feats_rl.push(req)                
                ## push dten
                buffcomm_feats_rl.push(dten_)
                buffcomm_feats_size_rl.push(out_size)
                    
            buffcomm_iter_rl.push(lim)

            #message(rank, "Time for gather 2: {:0.4f}",(time.time() - tic))
            prof.append('Gather II: {:0.4f}'.format(time.time() - tic))
            
            if epoch >= 2*nrounds_update or nrounds == 1:
                ticg = time.time()
                    
                lim = buffcomm_iter_rl.pop()                    
                for i in range(lim):
                    tic = time.time()
                    
                    req = gfqueue_feats_rl.pop(); req.wait()

                    #message(rank, "Time for async comms II: {:4f}", (time.time() - tic))
                    prof.append('Async comms II: {:0.4f}'.format(time.time() - tic))
                    
                    otf = buffcomm_feats_rl.pop()
                    out_size = buffcomm_feats_size_rl.pop()
                    stn = buffcomm_snl_lr.pop()
                    
                    offset = 0
                    for l in range(num_parts):
                        assert out_size[l] / (feat_size + 1) == stn[l].shape[0]
                        scatter_reduce_rl(otf, offset, stn[l], stn[l].shape[0],
                                          in_degs,
                                          feat, node_map_t, out_size[l], feat_size,
                                          num_parts, rank)
            
                        offset += out_size[l]

                #message(rank, "Time for scatter 2: {:0.4f}, roundn: {}", time.time() - ticg, roundn)
                prof.append('Scatter II: {:0.4f}'.format(time.time() - ticg))

        if rank == 0:
            print(prof, flush=True)
            print()
                
        return feat


    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None, None, None, None, None,\
            None, None, None, None, None, None, None


    
def call_drpa_core(neigh, adj,inner_node, lftensor, selected_nodes, node_map, num_parts,
                   rank, epoch, dist, degs, nrounds, output_sr_ar, gather_q41):
    return DRPACore.apply(neigh, adj, inner_node, lftensor,
                          selected_nodes, node_map, num_parts, rank,
                          epoch, dist, degs, nrounds, output_sr_ar, gather_q41)



class degree_division(torch.autograd.Function):
    #def __init__():

    @staticmethod
    def forward(ctx, neigh, h, feat_size, degs, lim):
        #print("in forward pass of deg_div", flush=True)
        deg_div(neigh, h, feat_size, degs, lim)
        ctx.backward_cache = feat_size, lim
        ctx.save_for_backward(degs)
        return neigh
    
    @staticmethod
    def backward(ctx, grad_out):
        #print("in backward pass of deg_div", flush=True)
        
        feat_size, lim = ctx.backward_cache
        degs = ctx.saved_tensors[0]
        deg_div_back(grad_out, feat_size, degs, lim)

        return grad_out, None, None, None, None

    
def deg_div_class(neigh, h, feat_size, degs, lim):
    return degree_division.apply(neigh, h, feat_size, degs, lim)
