import torch

from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from model.gumbel_softmax import masked_gumbel_softmax
from model.layers.graphsage import BatchedGraphSAGE
from model.model_utils import construct_mask, masked_softmax, BatchedTrace


class Assignment(nn.Module):
    def forward(self, x, adj, mask=None, log=False):
        raise NotImplementedError


class DiffPoolAssignment(Assignment):
    def __init__(self, nfeat, nnext, gumbel=False):
        super().__init__()
        self.gumbel = gumbel
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, mask=None, log=False):
        s_l_init = self.assign_mat(x, adj)
        if self.gumbel:
            s_l = masked_gumbel_softmax(s_l_init, None, temperature=0.5)
        else:
            s_l = F.softmax(s_l_init, dim=-1)
        return s_l


class MultihopDiffPoolAssignment(Assignment):
    def __init__(self, nfeat, nnext, gumbel=False, refine=True,
                 softmax_in_refine=False):
        super().__init__()
        self.gumbel = gumbel
        nhid = nfeat // 2
        self.assign_mat = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign_mat_2 = BatchedGraphSAGE(nhid, nnext, use_bn=True)
        self.refine = refine
        self.softmax_in_refine=softmax_in_refine
        if self.refine:
            self.refine_op = ProxRefinement()

    def forward(self, x, adj, mask=None, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = self.assign_mat_2(s_l_init, adj)
        if self.refine:
            lap_list = []
            for b in range(adj.shape[0]):
                lap_mat = torch.diag(torch.sum(adj[b,...],dim=-1)) -\
                adj[b,...]
                lap_list.append(lap_mat.unsqueeze(0))
            lap = torch.cat(lap_list,dim=0)
            #print(lap.shape)
            #print(adj.shape)
            #raise NotImplementedError
            #lap = torch.diag(torch.sum(adj, dim=-1)) - adj
            s_l = self.refine_op(s_l, lap)
        if self.gumbel:
            s_l = masked_gumbel_softmax(s_l, None, temperature=0.5)
        else:
            s_l = F.softmax(s_l, dim=-1)
        return s_l

class GlobalInfoDiffPoolAssignment(Assignment):
    def __init__(self, nfeat, nnext, gumbel=False, refine=True):
        super().__init__()
        self.gumbel = gumbel
        nhid = nfeat
        self.assign_mat = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign_mat_2 = BatchedGraphSAGE(nhid*2, nnext, use_bn=True)
        self.refine = refine
        if self.refine:
            self.refine_op = ProxRefinement()

    def forward(self, x, adj, mask=None, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l_init_sum = torch.mean(s_l_init, dim=-2).unsqueeze(1).repeat(1,
                                                                        s_l_init.shape[1],1)
        #print(s_l_init.shape)
        #print(s_l_init_sum.shape)
        #test = torch.cat((s_l_init, s_l_init_sum), dim=-1)
        #print(test.shape)
        #raise NotImplementedError
        s_l = self.assign_mat_2(torch.cat((s_l_init, s_l_init_sum), dim=-1), adj)

        if self.refine:
            lap_list = []
            for b in range(adj.shape[0]):
                lap_mat = torch.diag(torch.sum(adj[b,...],dim=-1)) -\
                adj[b,...]
                lap_list.append(lap_mat.unsqueeze(0))
            lap = torch.cat(lap_list,dim=0)
            #print(lap.shape)
            #print(adj.shape)
            #raise NotImplementedError
            #lap = torch.diag(torch.sum(adj, dim=-1)) - adj
            s_l = self.refine_op(s_l, lap)
            print("Finsish refine")
        if self.gumbel:
            s_l = masked_gumbel_softmax(s_l, None, temperature=0.5)
        else:
            s_l = F.softmax(s_l_init, dim=-1)
        return s_l



class RoutingAssignment(Assignment):
    # May not compatible with Entropy loss because of masking
    def __init__(self, nfeat, nnext, rtn):
        super(RoutingAssignment, self).__init__()
        self.routing_num = rtn
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, mask=None, log=False):
        # node_num = mask.sum(1)
        # cluster_num = (node_num * self.pool_ratio).long().to(x.device)
        # max_cluster_num = cluster_num.max()
        # cluster_mask = construct_mask(cluster_num, max_cluster_num).to(x.device)

        # z_l = self.embed(x, adj)

        # s_l = torch.randn(x.shape[0], x.shape[1], self.pool_size).to(x.device)
        s_l = self.assign_mat(x, adj)
        # if self.s_l is None:
        #     self.s_l = torch.randn(x.shape[0], x.shape[1], max_cluster_num).to(self.device)
        # else:
        #     self.s_l = self.s_l.clone().detach()

        # s_l = s_l

        for i in range(self.routing_num):
            # s_l = masked_softmax(s_l, cluster_mask, dim=-1)
            s_l = F.softmax(s_l, dim=-1)
            # if log:
            #     self.log[f's_{i}'] = s_l.cpu().numpy()
            xnext = self.squash(torch.matmul(s_l.transpose(-1, -2), x))
            delta_s_l = x @ xnext.transpose(-1, -2)
            s_l = s_l + delta_s_l

        s_l = F.softmax(s_l, dim=-1)
        return s_l

    @staticmethod
    def squash(s, dim=2):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s ** 2, dim=dim, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s


class RoutingAssignmentV2(Assignment):
    # May not compatible with Entropy loss because of masking

    def forward(self, x, adj, mask=None, log=False):
        node_num = mask.sum(1)
        cluster_num = (node_num * self.pool_ratio).long().to(x.device)
        max_cluster_num = cluster_num.max()
        cluster_mask = construct_mask(cluster_num, max_cluster_num).to(x.device)

        z_l = self.embed(x, adj)

        s_l = torch.randn(x.shape[0], x.shape[1], max_cluster_num).to(x.device)
        # if self.s_l is None:
        #     self.s_l = torch.randn(x.shape[0], x.shape[1], max_cluster_num).to(self.device)
        # else:
        #     self.s_l = self.s_l.clone().detach()

        s_l = s_l

        for i in range(self.routing_num):
            s_l = masked_softmax(s_l, cluster_mask, dim=-1)
            # if log:
            #     self.log[f's_{i}'] = s_l.cpu().numpy()
            xnext = self.squash(torch.matmul(s_l.transpose(-1, -2), z_l))
            delta_s_l = z_l @ xnext.transpose(-1, -2)
            s_l = s_l + delta_s_l

        s_l = masked_softmax(s_l, cluster_mask, dim=-1)
        return s_l

    @staticmethod
    def squash(s, dim=2):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s ** 2, dim=dim, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

class ProxRefinement(nn.Module):
    def __init__(self, aux_type="L1", reg=1e-1, gumbel=False):
        super().__init__()
        self.aux_type = aux_type
        self.gumbel = gumbel
        self.s0 = None
        self.soft_threshold = torch.nn.Softshrink(lambd=reg)
        self.softmax = nn.Softmax(dim=-1)
        self.batched_trace = BatchedTrace.apply

    def forward(self, s, lap, step=10, lr=1e-2, with_softmax=False):
        self.s0 = s
        reg = 10
        for _ in range(step):
            if not with_softmax:
                gradient = torch.matmul(torch.transpose(lap,-1,-2), s) +\
                torch.matmul(lap, s) - reg*2*(self.s0 - s)
                #print("##########")
                #print("reg mat gradient is ", 2*(self.s0 - s))
                s = self.soft_threshold(s - lr*gradient)
            else:
                gradient = self.find_grad(s, self.s0, lap)
                s = self.soft_threshold(s - lr*gradient)
            # add softmax?
            #s = self.softmax(s)

            # for test only
            #cut = torch.trace((torch.matmul(torch.transpose(s, -1, -2),\
            #                                torch.matmul(lap,s))))
            #cut = torch.sum(cut)
            #print("cut is ...", cut)
            #curr_score = torch.trace(torch.mm(torch.transpose(s, -1, -2),\
            #                                  torch.mm(lap, s)))\
            #        + torch.norm(s - self.s0)**2 + torch.norm(s, p=1)
            #print("distance with the original s0 is", torch.norm(s -
            #                                                     self.s0)**2)
            #print("current score is", curr_score)
            #print(torch.argmax(s, dim=1))

        #s = self.softmax(s)
        #s = F.relu(s)
        #cut = torch.trace((torch.matmul(torch.transpose(s, -1, -2),\
        #                                torch.matmul(lap,s))))
        #print("cut is", cut)
        #curr_score = torch.trace(torch.matmul(torch.transpose(s, -1, -2),\
        #                                      torch.matmul(lap, s)))\
        #        + torch.norm(s - self.s0) + torch.norm(s, p=1)
        #print("curr score is", curr_score)
        #print(s)
        #print(torch.argmax(s, dim=1))
        #print(torch.sum(s, dim=-1))

        #if self.gumbel:
            #pass
        #    s = masked_gumbel_softmax(s, None, temperature=0.5)
        #else:
            #pass
        #    s = F.softmax(s, dim=-1)

        return s

    def find_grad(self, s, s0, lap):
        with torch.enable_grad():
            s_cp = s.clone().detach().requires_grad_(True)
            print(s_cp.shape)
            s0_cp = s0.clone().detach().requires_grad_(True)
            print(lap.shape)
            cut =\
            torch.trace(torch.matmul(torch.transpose(self.softmax(s_cp),-1,-2),torch.matmul(lap,\
                                                                                     self.softmax(s_cp))))
            norm_reg = torch.pow(torch.norm(s0_cp-s_cp, dim=(-1,-2)),2)
            val = cut + norm_reg
            val.backward()

        return s_cp.grad

#
# class BatchedRandomAdjPool(BatchedDiffPool):
#     # def __init__(self, nfeat, nnext, nhid, device='cpu', link_pred=False):
#     #     super().__init__(nfeat, nnext, nhid, device, link_pred)
#
#     def forward(self, x, adj, mask=None, log=False):
#         z_l = self.embed(x, adj)
#         s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
#         if log:
#             self.log['s'] = s_l.cpu().numpy()
#         xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
#         # anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
#         anext = torch.eye(s_l.size(2)).to(self.device).unsqueeze(0).expand(z_l.size(0), -1, -1)
#         if self.link_pred:
#             self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
#             self.link_pred_loss = self.link_pred_loss / (adj.size(1) * adj.size(2))
#             self.entropy_loss = torch.distributions.Categorical(probs=s_l).entropy()
#             if mask is not None:
#                 self.entropy_loss = self.entropy_loss * mask.expand_as(self.entropy_loss)
#             self.entropy_loss = self.entropy_loss.sum(-1)
#             self.entropy_loss = self.entropy_loss.sum(-1) / (adj.size(1) * adj.size(2))
#         if log:
#             self.log['a'] = anext.cpu().numpy()
#         return xnext, anext
