# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT
import numpy as np

def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = th.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res

def batched_l1_dist(a, b):
    res = th.cdist(a, b, p=1)
    return res

class TransEScore(nn.Module):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """
    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else: # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=self.dist_ord, dim=-1)}

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, chunk_size, hidden_dim)
                return gamma - self.neg_dist_func(tails, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                return gamma - self.neg_dist_func(heads, tails)
            return fn

class TransRScore(nn.Module):
    """TransR score function
    Paper link: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523
    """
    def __init__(self, gamma, projection_emb, relation_dim, entity_dim):
        super(TransRScore, self).__init__()
        self.gamma = gamma
        self.projection_emb = projection_emb
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.data['head_emb']
        tail = edges.data['tail_emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def prepare(self, g, gpu_id, trace=False):
        head_ids, tail_ids = g.all_edges(order='eid')
        projection = self.projection_emb(g.edata['id'], gpu_id, trace)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        g.edata['head_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][head_ids], projection)
        g.edata['tail_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][tail_ids], projection)

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(num_chunks, -1, 1, self.entity_dim)
                tail = th.matmul(tail, projection)
                tail = tail.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                head = head.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                head = th.matmul(head, projection)
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                head = head.reshape(num_chunks, -1, 1, self.entity_dim)
                head = th.matmul(head, projection)
                head = head.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                tail = tail.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                tail = th.matmul(tail, projection)
                return head, tail
            return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def reset_parameters(self):
        self.projection_emb.init(1.0)

    def update(self, gpu_id=-1):
        self.projection_emb.update(gpu_id)

    def save(self, path, name):  
        self.projection_emb.save(path, name+'projection')

    def load(self, path, name):
        self.projection_emb.load(path, name+'projection')

    def prepare_local_emb(self, projection_emb):
        self.global_projection_emb = self.projection_emb
        self.projection_emb = projection_emb

    def prepare_cross_rels(self, cross_rels):
        self.projection_emb.setup_cross_rels(cross_rels, self.global_projection_emb)

    def writeback_local_emb(self, idx):
        self.global_projection_emb.emb[idx] = self.projection_emb.emb.cpu()[idx]

    def load_local_emb(self, projection_emb):
        device = projection_emb.emb.device
        projection_emb.emb = self.projection_emb.emb.to(device)
        self.projection_emb = projection_emb

    def share_memory(self):
        self.projection_emb.share_memory()  

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                heads = heads - relations
                heads = heads.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn

class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)
            return fn

class ComplExScore(nn.Module):
    """ComplEx score function
    Paper link: https://arxiv.org/abs/1606.06357
    """
    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, -1)}

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                return th.bmm(tmp, tails)
            return fn

class RESCALScore(nn.Module):
    """RESCAL score function
    Paper link: http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """
    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        score = head * th.matmul(rel, tail).squeeze(-1)
        # TODO: check if use self.gamma
        return {'score': th.sum(score, dim=-1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tails = tails.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, tails).squeeze(-1)
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                heads = heads.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, heads).squeeze(-1)
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)
            return fn

class RotatEScore(nn.Module):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """
    def __init__(self, gamma, emb_init):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)

        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))
        
    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn
    
    def prepare(self, g, gpu_id, trace=False):
        pass
    
    def create_neg(self, neg_head):
        gamma = self.gamma
        emb_init = self.emb_init
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - heads
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
                return gamma - score.sum(-1)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real

                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - tails
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

                return gamma - score.sum(-1)

            return fn
