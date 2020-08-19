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

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd

def batched_l2_dist(a, b):
    a_squared = nd.power(nd.norm(a, axis=-1), 2)
    b_squared = nd.power(nd.norm(b, axis=-1), 2)

    squared_res = nd.add(nd.linalg_gemm(
        a, nd.transpose(b, axes=(0, 2, 1)), nd.broadcast_axes(nd.expand_dims(b_squared, axis=-2), axis=1, size=a.shape[1]), alpha=-2
    ), nd.expand_dims(a_squared, axis=-1))
    res = nd.sqrt(nd.clip(squared_res, 1e-30, np.finfo(np.float32).max))
    return res

def batched_l1_dist(a, b):
    a = nd.expand_dims(a, axis=-2)
    b = nd.expand_dims(b, axis=-3)
    res = nd.norm(a - b, ord=1, axis=-1)
    return res

class TransEScore(nn.Block):
    """ TransE score function
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
        return {'score': self.gamma - nd.norm(score, ord=self.dist_ord, axis=-1)}

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

class TransRScore(nn.Block):
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
        return {'score': self.gamma - nd.norm(score, ord=1, axis=-1)}

    def prepare(self, g, gpu_id, trace=False):
        head_ids, tail_ids = g.all_edges(order='eid')
        projection = self.projection_emb(g.edata['id'], gpu_id, trace)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        head_emb = g.ndata['emb'][head_ids.as_in_context(g.ndata['emb'].context)].expand_dims(axis=-2)
        tail_emb = g.ndata['emb'][tail_ids.as_in_context(g.ndata['emb'].context)].expand_dims(axis=-2)
        g.edata['head_emb'] = nd.batch_dot(head_emb, projection).squeeze()
        g.edata['tail_emb'] = nd.batch_dot(tail_emb, projection).squeeze()

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(-1, 1, self.entity_dim)
                tail = nd.batch_dot(tail, projection)
                tail = tail.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                head = head.reshape(num_chunks, -1, 1, self.entity_dim)
                num_rels = projection.shape[1]
                num_nnodes = head.shape[1]

                heads = []
                for i in range(num_chunks):
                    head_negs = []
                    for j in range(num_nnodes):
                        head_neg = head[i][j]
                        head_neg = head_neg.reshape(1, 1, self.entity_dim)
                        head_neg = nd.broadcast_axis(head_neg, axis=0, size=num_rels)
                        head_neg = nd.batch_dot(head_neg, projection[i])
                        head_neg = head_neg.squeeze(axis=1)
                        head_negs.append(head_neg)
                    head_negs = nd.stack(*head_negs, axis=1)
                    heads.append(head_negs)
                head = nd.stack(*heads)
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
                head = head.reshape(-1, 1, self.entity_dim)
                head = nd.batch_dot(head, projection).squeeze()
                head = head.reshape(num_chunks, -1, self.relation_dim)

                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(num_chunks, -1, 1, self.entity_dim)
                num_rels = projection.shape[1]
                num_nnodes = tail.shape[1]

                tails = []
                for i in range(num_chunks):
                    tail_negs = []
                    for j in range(num_nnodes):
                        tail_neg = tail[i][j]
                        tail_neg = tail_neg.reshape(1, 1, self.entity_dim)
                        tail_neg = nd.broadcast_axis(tail_neg, axis=0, size=num_rels)
                        tail_neg = nd.batch_dot(tail_neg, projection[i])
                        tail_neg = tail_neg.squeeze(axis=1)
                        tail_negs.append(tail_neg)
                    tail_negs = nd.stack(*tail_negs, axis=1)
                    tails.append(tail_negs)
                tail = nd.stack(*tails)
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

    def writeback_local_emb(self, idx):
        self.global_projection_emb.emb[idx] = self.projection_emb.emb.as_in_context(mx.cpu())[idx]

    def load_local_emb(self, projection_emb):
        context = projection_emb.emb.context
        projection_emb.emb = self.projection_emb.emb.as_in_context(context)
        self.projection_emb = projection_emb

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - nd.norm(score, ord=1, axis=-1)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                heads = heads - relations
                heads = heads.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - nd.norm(score, ord=1, axis=-1)
            return fn

class DistMultScore(nn.Block):
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
        return {'score': nd.sum(score, axis=-1)}

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
                heads = nd.transpose(heads, axes=(0, 2, 1))
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = nd.transpose(tails, axes=(0, 2, 1))
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, tails)
            return fn

class ComplExScore(nn.Block):
    """ComplEx score function
    Paper link: https://arxiv.org/abs/1606.06357
    """
    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = nd.split(edges.src['emb'], num_outputs=2, axis=-1)
        real_tail, img_tail = nd.split(edges.dst['emb'], num_outputs=2, axis=-1)
        real_rel, img_rel = nd.split(edges.data['emb'], num_outputs=2, axis=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': nd.sum(score, -1)}

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
                emb_real, emb_img = nd.split(tails, num_outputs=2, axis=-1)
                rel_real, rel_img = nd.split(relations, num_outputs=2, axis=-1)
                real = emb_real * rel_real + emb_img * rel_img
                img = -emb_real * rel_img + emb_img * rel_real
                emb_complex = nd.concat(real, img, dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = nd.transpose(heads, axes=(0, 2, 1))
                return nd.linalg_gemm2(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real, emb_img = nd.split(heads, num_outputs=2, axis=-1)
                rel_real, rel_img = nd.split(relations, num_outputs=2, axis=-1)
                real = emb_real * rel_real - emb_img * rel_img
                img = emb_real * rel_img + emb_img * rel_real
                emb_complex = nd.concat(real, img, dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)

                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = nd.transpose(tails, axes=(0, 2, 1))
                return nd.linalg_gemm2(tmp, tails)
            return fn

class RESCALScore(nn.Block):
    """RESCAL score function
    Paper link: http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """
    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].expand_dims(2)
        rel = edges.data['emb']
        rel = rel.reshape(-1, self.relation_dim, self.entity_dim)
        score = head * mx.nd.batch_dot(rel, tail).squeeze()
        # TODO: check if use self.gamma
        return {'score': mx.nd.sum(score, -1)}
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
                heads = mx.nd.transpose(heads, axes=(0,2,1))
                tails = tails.expand_dims(2)
                relations = relations.reshape(-1, self.relation_dim, self.entity_dim)
                tmp = mx.nd.batch_dot(relations, tails).squeeze()
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = mx.nd.transpose(tails, axes=(0,2,1))
                heads = heads.expand_dims(2)
                relations = relations.reshape(-1, self.relation_dim, self.entity_dim)
                tmp = mx.nd.batch_dot(relations, heads).squeeze()
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, tails)
            return fn

class RotatEScore(nn.Block):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """
    def __init__(self, gamma, emb_init, eps=1e-10):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init
        self.eps = eps

    def edge_func(self, edges):
        real_head, img_head = nd.split(edges.src['emb'], num_outputs=2, axis=-1)
        real_tail, img_tail = nd.split(edges.dst['emb'], num_outputs=2, axis=-1)

        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = nd.cos(phase_rel), nd.sin(phase_rel)
        real_score = real_head * re_rel - img_head * im_rel
        img_score = real_head * im_rel + img_head * re_rel
        real_score = real_score - real_tail
        img_score = img_score - img_tail
        #sqrt((x*x).sum() + eps)
        score = mx.nd.sqrt(real_score * real_score + img_score * img_score + self.eps).sum(-1)
        return {'score': self.gamma - score} 

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
        gamma = self.gamma
        emb_init = self.emb_init
        eps = self.eps
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real, emb_img = nd.split(tails, num_outputs=2, axis=-1)
                phase_rel = relations / (emb_init / np.pi)

                rel_real, rel_img = nd.cos(phase_rel), nd.sin(phase_rel)
                real = emb_real * rel_real + emb_img * rel_img
                img = -emb_real * rel_img + emb_img * rel_real
                emb_complex = nd.concat(real, img, dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)

                score = tmp - heads
                score_real, score_img = nd.split(score, num_outputs=2, axis=-1)
                score = mx.nd.sqrt(score_real * score_real + score_img * score_img + self.eps).sum(-1)
 
                return gamma - score
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real, emb_img = nd.split(heads, num_outputs=2, axis=-1)
                phase_rel = relations / (emb_init / np.pi)

                rel_real, rel_img = nd.cos(phase_rel), nd.sin(phase_rel)
                real = emb_real * rel_real - emb_img * rel_img
                img = emb_real * rel_img + emb_img * rel_real
                emb_complex = nd.concat(real, img, dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)

                score = tmp - tails
                score_real, score_img = nd.split(score, num_outputs=2, axis=-1)
                score = mx.nd.sqrt(score_real * score_real + score_img * score_img + self.eps).sum(-1)
 
                return gamma - score
            return fn


