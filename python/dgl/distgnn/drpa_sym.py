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

from ..sparse import scatter_reduce_v41, fdrpa_gather_emb_v41, scatter_reduce_v42, fdrpa_gather_emb_v42, fdrpa_get_buckets_v4, deg_div, deg_div_back, fdrpa_init_buckets_v4

display = False
class gqueue():
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

gfqueue  = gqueue()
gfqueue2 = gqueue()
gfqueue_fp2 = gqueue()
gbqueue  = gqueue()

## sym
fstorage_comm_feats_async  = gqueue()
fstorage_comm_feats_async2  = gqueue()
fstorage_comm_feats_chunk_async  = gqueue()
fstorage_comm_feats_chunk_async2  = gqueue()
fstorage_comm_nodes_async  = gqueue()
fstorage_comm_nodes_async2 = gqueue()
fstorage_comm_nodes_async_fp2 = gqueue()
fstorage_comm_nodes_async_fp22 = gqueue()

fstorage_comm_iter = gqueue()
fp2_fstorage_comm_feats_async = gqueue()
fp2_fstorage_comm_feats_chunk_async = gqueue()
fp2_fstorage_comm_iter = gqueue()
fstorage_comm_nodes_async_fp22_ext = gqueue()


class drpa_:
    ## static variables
    rank = -1
    num_parts = 0
    node_map = 0
    nrounds = 0
    output_sr_ar = []
    selected_nodes = []
    epochs_ar = []
    epochi = 0
    nlayer = 0
    dist = 0
    gather_q41 = 0
    
    @staticmethod
    def drpa_initialize(g, rank, num_parts, node_map, nrounds, dist, nlayers):
        drpa.rank = rank
        drpa.num_parts = num_parts
        drpa.node_map = node_map
        drpa.nrounds = nrounds
        drpa.dist = dist

        drpa.epochs_ar = [0 for i in range(nlayers)]
        drpa.epochi = 0
        drpa.nalyers = nlayers
        drpa.gather_q41 = gqueue()
        drpa.output_sr_ar = []
        
        ## Creates buckets based on ndrounds
        drpa.drpa_create_buckets(g)

        adj = g.dstdata['adj']
        lf = g.dstdata['lf']
        width = adj.shape[1]
        
        ## groups send data according to 'send to rank'
        ## communicates and fill output_sr_ar for data to be recd.
        drpa.drpa_init_buckets(g, adj, lf, drpa.selected_nodes, num_parts,
                               nrounds, rank, node_map, width)

    @staticmethod
    def drpa_update_all(self, g, message_func, reduce_func, apply_node_func):
        assert rank != -1, "drpa not initialized !!!"
        neigh = g.dstdata['neigh']
        adj = g.dstdata['adj']
        inner_node = g.dstdata['inner_node']
        lftensor = g.dstdata['lf']

        epoch = drpa.epochs_ar[epoci]
        
        in_degs = g.in_degrees().to(feat_dst)
        self.dstdata['neigh'] = call_drpa_core(neigh, adj, inner_node, lftensor, drpa.selected_nodes,
                                               drpa.node_map, drpa.num_parts, epoch, drpa.rank,
                                               in_degs,
                                               drpa.output_sr_ar,
                                               drpa.gather_q41)
        
        drpa.epochs_ar[epochi]
        epochi = (epochi + 1) % drpa.nlayers
        
        self.ndata['par_in_degs'] = in_degs
        #return in_degs
    

    @staticmethod
    def drpa_init_buckets(g, adj, lf, selected_nodes, num_parts, nrounds, rank, node_map, width):

        for l in range(nrounds):
            buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
            ## fill buckets here
            fdrpa_init_buckets_v4(adj, selected_nodes[l], node_map, buckets,
                                  lf, width, drpa.num_parts, rank)
            input_sr = []
            for i in range(0, num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))
                
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async            
            sync_req.wait() ## recv the #nodes communicated
            drpa.output_sr_ar.append(output_sr)  ## output


        
    @staticmethod
    def drpa_create_buckets(g, nrounds, rank):
        inner_nodex = g.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)
        
        nrounds_ = nrounds
        drpa_relay.selected_nodes = [ [] for i in range(nrounds_)]  ## #native nodes

        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = inner_nodex.count(0)  ## count split nodes
        alien_nodes_per_round = int((total_alien_nodes + nrounds_ -1) / nrounds_)
        
        counter = 0
        pos = 0
        r = 0
        while counter < n:
            if inner_nodex[counter] == 0:    ##split node
                drpa_relay.selected_nodes[r].append(counter)
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
        
        #return ## selected_nodes
        if rank == 0:
            print("Selected nodes in each round: ", flush=True)
            for i in range(nrounds_):
                print("round: ", i,  " nodes: ", len(drpa_relay.selected_nodes[i]), flush=True);


## DRPA
def drpa(gobj, rank, num_parts, node_map, nrounds, dist, nlayers):
    #d = drpa_master(gobj, rank, num_parts, node_map, nrounds, dist, nlayers)
    d = drpa_master(gobj._graph, gobj._ntypes, gobj._etypes, gobj._node_frames, gobj._edge_frames)
    d.drpa_init(rank, num_parts, node_map, nrounds, dist, nlayers)
    return d
    
class drpa_master(DGLHeteroGraph):
    
        
    def drpa_init(self, rank, num_parts, node_map, nrounds, dist, nlayers):
        #print("In drpa Init....")
        self.rank = rank
        self.num_parts = num_parts
        self.node_map = node_map
        self.nrounds = nrounds
        self.dist = dist
        self.nlayers = nlayers + 1

        self.epochs_ar = [0 for i in range(self.nlayers)]
        self.epochi = 0
        self.gather_q41 = gqueue()
        self.output_sr_ar = []

        if self.nrounds == -1: return
        ## Creates buckets based on ndrounds        

        adj = self.dstdata['adj']
        lf = self.dstdata['lf']
        width = adj.shape[1]
        #print("width: ", width)
        ## groups send data according to 'send to rank'
        ## communicates and fill output_sr_ar for data to be recd.
        self.drpa_create_buckets()
        self.drpa_init_buckets(adj, lf, width)

        
    def drpa_finalize(self):
        if self.rank == 0:
            print("Symm: Clearning backlogs", flush=True)
        while gfqueue.empty() == False:
            req = gfqueue.pop()
            req.wait()
            req = gfqueue2.pop()
            req.wait()
        while gfqueue_fp2.empty() == False:                
            req = gfqueue_fp2.pop()
            req.wait()

        fstorage_comm_feats_async.purge()
        fstorage_comm_feats_async2.purge()
        fstorage_comm_feats_chunk_async.purge()
        fstorage_comm_feats_chunk_async2.purge()
        fstorage_comm_nodes_async_fp2.purge()
        fstorage_comm_nodes_async_fp22.purge()
        fstorage_comm_iter.purge()
        fp2_fstorage_comm_feats_async.purge()
        fp2_fstorage_comm_feats_chunk_async.purge()
        fp2_fstorage_comm_iter.purge()
        

        if gfqueue.empty() == False:
            print("gfqueue not empty after backlogs flushing", flush=True)
        if gfqueue2.empty() == False:
            print("gfqueue2 not empty after backlogs flushing", flush=True)
        if gfqueue_fp2.empty() == False:
            print("gfqueue_fp2 not empty after backlogs flushing", flush=True)
            
        if self.rank == 0:
            print("Clearning backlogs ends ", flush=True)

        
    #@staticmethod ## overload
    ## not used for now
    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        assert self.rank != -1, "drpa not initialized !!!"

        #print("rfunc: ", reduce_func.name)
        mean = 0
        if reduce_func.name == "mean":
            reduce_func = fn.sum('m', 'neigh')
            mean = 1
            
        #print("rfunc: ", reduce_func.name)
        
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
        self.dstdata['neigh'] = call_drpa_core(neigh, adj, inner_node, lftensor, self.selected_nodes,
                                               self.node_map, self.num_parts, self.rank, epoch,
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
            ## fill buckets here
            #print("nrounds: ", self.nrounds)
            #print("l: ", l)
            #print("selected_nodes: ", len(self.selected_nodes[l]))
            sn = torch.tensor(self.selected_nodes[l], dtype=torch.int32)
            nm = torch.tensor(self.node_map, dtype=torch.int32)
            fdrpa_init_buckets_v4(adj, sn, nm, buckets,
                                  lf, width, self.num_parts, self.rank)
            input_sr = []
            for i in range(0, self.num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))
                
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, self.num_parts)]
            sync_req = self.dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated
            self.output_sr_ar.append(output_sr)  ## output


    def drpa_create_buckets(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)
        
        nrounds_ = self.nrounds
        self.selected_nodes = [ [] for i in range(nrounds_)]  ## #native nodes

        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = inner_nodex.count(0)  ## count split nodes
        alien_nodes_per_round = int((total_alien_nodes + nrounds_ -1) / nrounds_)
        
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
        #    for i in range(nrounds_):
        #        print("round: ", i,  " nodes: ", len(self.selected_nodes[i]), flush=True);

                
    def in_degrees(self):
        try:
            return self.r_in_degs
        except:
            print("Passing only local node degrees.")
            pass
    
        return DGLHeteroGraph.in_degrees(self)

    
class drpa_core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_, adj_, inner_node_, lftensor_, selected_nodes,
                node_map, num_parts, rank, epoch, dist, in_degs, nrounds,
                output_sr_ar, gather_q41):        
        nrounds_update = nrounds
        
        feat = feat_
        adj = adj_
        inner_node = inner_node_
        lftensor = lftensor_
        feat_size = feat.shape[1]
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        base_chunk_size = int(int_threshold / num_parts) ##bytes            
        base_chunk_size_fs = floor(base_chunk_size / (feat_size + 1) )
        node_map_t = torch.tensor(node_map, dtype=torch.int32)
        selected_nodes_t = []
        for sn in selected_nodes:
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))
            
        roundn =  epoch % nrounds
            
        ## section I: prepare the msg
        buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        width = adj.shape[1]

        num_sel_nodes = torch.tensor([0], dtype=torch.int32)
        cnt = 0

        if rank == 0 and display:
            print("FWD pass: nrounds: ", nrounds, flush=True)
            
        if rank == 0:
            tic = time.time()

        #### 1. get bucket sizes
        node2part = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        node2part_index = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        fdrpa_get_buckets_v4(adj, selected_nodes_t[roundn], node2part, node2part_index,
                             node_map_t, buckets, lftensor, num_sel_nodes,
                             width, num_parts, rank)

        
        if rank == 0  and display:
            toc = time.time()
            print("Time for get buckets: {:0.4f} in epoch: {}".format(toc - tic, epoch))
        
            
        ###### comms to gather the bucket sizes for all-to-all feats
        input_sr = []
        #output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
        for i in range(0, num_parts):
            input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))


        if False:
            output_sr = output_sr_ar[roundn].copy()   ## computed during drpa_init             
        else:
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
        #if rank == 0 and display:
        tic = time.time()
            
        cum = 0
        flg = 0
        for i in output_sr:
            cum += int(i) * (feat_size + 1)
            if int(i) >= base_chunk_size_fs: flg = 1

        for i in input_sr:
            if int(i) >= base_chunk_size_fs: flg = 1
        
        back_off = 1
        if cum >= int_threshold or send_feat_len >= int_threshold or flg:        
            for i in range(num_parts):
                val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                if val > back_off:
                    back_off = val
                    
        #print("back_off: {}".format(back_off), flush=True)
        tback_off = torch.tensor(back_off)
        rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, async_op=True)
        
        #lim = int(tback_off)  ## async req waited at the bottom of the loop
        lim = 1            
        soffset_base = [0 for i in range(num_parts)]  ## min chunk size        
        soffset_cur = [0 for i in range(num_parts)]       ## send list of ints
        roffset_cur = [0 for i in range(num_parts)]       ## recv list of intrs
        
        j=0
        while j < lim:
            tsend = 0
            trecv = 0
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

            send_node_list  = [torch.empty(soffset_cur[i], dtype=torch.int32) for i in range(num_parts)]
            sten_ = torch.empty(tsend * (feat_size + 1), dtype=feat.dtype)
            dten_ = torch.empty(trecv * (feat_size + 1), dtype=feat.dtype)
            sten_nodes_ = torch.empty(tsend , dtype=torch.int32)
            dten_nodes_ = torch.empty(trecv , dtype=torch.int32)
            
            out_size = [0 for i in range(num_parts)]
            in_size = [0 for i in range(num_parts)]
            out_size_nodes = [0 for i in range(num_parts)]
            in_size_nodes = [0 for i in range(num_parts)]

            if rank == 0:
                gtic = time.time()
                
            offset = 0            
            for i in range(num_parts):
                ## gather by followers
                fdrpa_gather_emb_v41(feat, feat.shape[0], adj, sten_, offset,
                                     send_node_list[i], sten_nodes_,
                                     selected_nodes_t[roundn],
                                     in_degs, node2part, node2part_index, width, feat_size, i,
                                     soffset_base[i], soffset_cur[i], node_map_t, num_parts)
                
                out_size[i]       = roffset_cur[i] * (feat_size + 1)
                in_size[i]        = soffset_cur[i] * (feat_size + 1)
                offset            += soffset_cur[i]
                out_size_nodes[i] = roffset_cur[i]
                in_size_nodes[i]  = soffset_cur[i]


            if rank == 0:
                gtoc = time.time()
    
            if rank == 0 and display:
                print("Sending {}, recving {} data I".format(tsend, trecv), flush=True)
                
            req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)    	
            gfqueue.push(req)
            req2 = dist.all_to_all_single(dten_nodes_, sten_nodes_,
                                         out_size_nodes, in_size_nodes,
                                         async_op=True)    	
            gfqueue2.push(req2)

                
            soffset_cur_copy = soffset_cur.copy()

    	    ## section III: store pointers for the data in motion
            fstorage_comm_feats_async.push(dten_)
            fstorage_comm_feats_async2.push(dten_nodes_)
            fstorage_comm_feats_chunk_async.push(out_size)
            fstorage_comm_feats_chunk_async2.push(out_size_nodes)
            fstorage_comm_nodes_async_fp2.push(send_node_list)    ## fwd phase II
            fstorage_comm_nodes_async_fp22.push(soffset_cur_copy) ## fwd phase II
            
            if j == 0:
                rreq.wait()
                lim = int(tback_off)
                #print("lim: ", lim)
            j += 1
        ##############################################################################        
        ## mpi call split ends
        if rank == 0  and display:
            print("Max iters in MPI split comm: {}".format(lim), flush=True)
            
        fstorage_comm_iter.push(lim)
        
        toc = time.time()
        if rank == 0 and display:
            print("Time for Gather I: {:0.4f}".format(toc - tic), flush=True)
            
        
        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        
        if epoch >= nrounds_update or nrounds == 1:
            if gfqueue.empty() == True:
                print("Forward pass Error: epoch: ", epoch, " Empty queue !!!")

            #if rank == 0 and display:
            ticg = time.time()
                
            lim = fstorage_comm_iter.pop()
            out_size_nodes_ar = []                
            for i in range(lim):
                #print("i: ", i, flush=True)
                if rank == 0 and display:                
                    tic = time.time()

                req = gfqueue.pop()
                req.wait()
                req = gfqueue2.pop()
                req.wait()
                
                if rank == 0 and display:
                    print("Time for async comms I: {:4f}".format(time.time() - tic), flush=True)
                    
                otf = fstorage_comm_feats_async.pop()
                out_size = fstorage_comm_feats_chunk_async.pop()
                otn = fstorage_comm_feats_async2.pop()
                out_size_nodes = fstorage_comm_feats_chunk_async2.pop()
                out_size_nodes_ar.append(out_size_nodes)

                ilen = 0
                recv_list_nodes_ar = []
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))
            
                pos = torch.tensor([0], dtype=torch.int64)        
                offsetf = 0
                offsetn = 0
                for l in range(num_parts):
                    scatter_reduce_v41(otf, offsetf, otn, offsetn,
                                       feat, in_degs, node_map_t, out_size[l], feat_size,
                                       num_parts, recv_list_nodes_ar[l], pos,
                                       int(out_size_nodes[l]), rank)
                    
                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]
                    
                assert ilen == pos[0], "Issue in scatter reduce!"                
                recv_list_nodes.append(recv_list_nodes_ar)


            tocg = time.time()
            if rank == 0 and display:
                print("Time for scatter I: {:0.4f} in epoch: {}".format(tocg - ticg, epoch), flush=True)

            #if rank == 0 and display:
            tic = time.time()

            ### gather-scatter round II
            for j in range(lim):
                tsend = 0
                trecv = 0
                stn_fp2 = fstorage_comm_nodes_async_fp22.pop()
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
                    ## gather by leader
                    fdrpa_gather_emb_v42(feat, feat.shape[0], sten_, offset,
                                         recv_list_nodes_[i], out_size_nodes[i],
                                         in_degs,
                                         feat_size, i,
                                         node_map_t, num_parts)
                    
                    out_size[i]       = stn_fp2[i] * (feat_size + 1)
                    in_size[i]        = out_size_nodes[i] * (feat_size + 1)
                    offset           += in_size[i]
                                    
                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)    	
                gfqueue_fp2.push(req)
                
                ## push dten
                fp2_fstorage_comm_feats_async.push(dten_)
                fp2_fstorage_comm_feats_chunk_async.push(out_size)
                #dist.barrier()
                    
            fp2_fstorage_comm_iter.push(lim)

            toc = time.time()
            if rank == 0 and display:
                print("Time for gather 2: {:0.4f}".format(toc - tic), flush=True)
                
            
            if epoch >= 2*nrounds_update or nrounds == 1:
                ## No backpass for this one
                #if rank == 0 and display:
                ticg = time.time()
                    
                lim = fp2_fstorage_comm_iter.pop()
                    
                for i in range(lim):
                    #dist.barrier()
                    if rank == 0 and display:
                        tic = time.time()
                    
                    req = gfqueue_fp2.pop()
                    req.wait()
                    
                    if rank == 0 and display:
                        print("Time for async comms II: {:4f}".format(time.time() - tic), flush=True)
                    
                    otf = fp2_fstorage_comm_feats_async.pop()
                    out_size = fp2_fstorage_comm_feats_chunk_async.pop()
                    stn = fstorage_comm_nodes_async_fp2.pop()
                    
                    offset = 0
                    for l in range(num_parts):
                        assert out_size[l] / (feat_size + 1) == stn[l].shape[0]
                        scatter_reduce_v42(otf, offset, stn[l], stn[l].shape[0],
                                           in_degs,
                                           feat, node_map_t, out_size[l], feat_size,
                                           num_parts, rank)
            
                        offset += out_size[l]

                tocg = time.time()
                if rank == 0 and display:
                    print("Time for scatter 2: {:0.4f}".format(tocg - ticg))
                
        return feat


    @staticmethod
    def backward(ctx, grad_out):

        return grad_out, None, None, None, None, None, None, None, None, None, None, None, None, None


    
def call_drpa_core(neigh, adj,inner_node, lftensor, selected_nodes, node_map, num_parts,
                   rank, epoch, dist, degs, nrounds, output_sr_ar, gather_q41):
    return drpa_core.apply(neigh, adj, inner_node, lftensor,
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
