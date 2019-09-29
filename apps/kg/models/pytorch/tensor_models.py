"""
Knowledge Graph Embedding Models.
1. TransE
2. DistMult
3. ComplEx
4. RotatE
5. pRotatE
6. TransH
7. TransR
8. TransD
9. RESCAL
"""
import os
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT

from .score_fun import *
from .. import *

class ExternalEmbedding:
    def __init__(self, args, num, dim, device):
        self.gpu = args.gpu
        self.args = args
        self.trace = []

        self.emb = th.empty(num, dim, dtype=th.float32, device=device)
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()
        self.state_step = 0

    def init(self, emb_init):
        INIT.uniform_(self.emb, -emb_init, emb_init)
        INIT.zeros_(self.state_sum)

    def share_memory(self):
        self.emb.share_memory_()
        self.state_sum.share_memory_()

    def __call__(self, idx, gpu_id=-1, trace=True):
        s = self.emb[idx]
        if self.gpu >= 0:
            s = s.cuda(self.gpu)
        data = s.clone().detach().requires_grad_(True)
        if trace:
            self.trace.append((idx, data))
        return data

    def update(self):
        self.state_step += 1
        with th.no_grad():
            for idx, data in self.trace:
                grad = data.grad.data

                clr = self.args.lr
                #clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

                # the update is non-linear so indices must be unique
                grad_indices = idx
                grad_values = grad

                grad_sum = (grad_values * grad_values).mean(1)
                device = self.state_sum.device
                if device != grad_indices.device:
                    grad_indices = grad_indices.to(device)
                if device != grad_sum.device:
                    grad_sum = grad_sum.to(device)
                self.state_sum.index_add_(0, grad_indices, grad_sum)
                std = self.state_sum[grad_indices]  # _sparse_mask
                std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                if self.gpu >= 0:
                    std_values = std_values.cuda(self.args.gpu)
                tmp = (-clr * grad_values / std_values)
                if tmp.device != device:
                    tmp = tmp.to(device)
                # TODO(zhengda) the overhead is here.
                self.emb.index_add_(0, grad_indices, tmp)
        self.trace = []

    def curr_emb(self):
        data = [data for _, data in self.trace]
        return th.cat(data, 0)

    def save(self, path, name):
        file_name = os.path.join(path, name)
        np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        file_name = os.path.join(path, name)
        self.emb = th.Tensor(np.load(file_name))


class PBGKEModel(nn.Module):
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False):
        super(PBGKEModel, self).__init__()
        self.args = args
        self.n_entities = n_entities
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = 2.0
        self.emb_init = (gamma + self.eps) / hidden_dim

        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        device = th.device('cpu') if args.gpu < 0 else th.device('cuda:' + str(args.gpu))
        self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim,
                                            th.device('cpu') if args.mix_cpu_gpu else device)
        # For RESCAL, relation_emb = relation_dim * entity_dim
        if model_name == 'RESCAL':
            rel_dim = relation_dim * entity_dim
        else:
            rel_dim = relation_dim
        self.relation_emb = ExternalEmbedding(args, n_relations, rel_dim, device)

        if model_name == 'TransE':
            self.score_func = TransEScore(gamma)
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RotatE':
            self.score_func = RotatEScore()
        elif model_name == 'pRotatE':
            self.score_func = pRotatEScore()
        elif model_name == 'TransH':
            self.score_func = TransHScore()
        elif model_name == 'TransR':
            self.score_func = TransRScore()
        elif model_name == 'TransD':
            self.score_func = TransDScore()
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore()

        self.test_basic_models = {}
        if args.train:
            self.train_basic_model = BasePBGKEModel(self.score_func,
                                                    self.args.batch_size,
                                                    self.args.neg_sample_size,
                                                    hidden_dim,
                                                    n_entities)
        if args.valid:
            self.test_basic_models[self.args.neg_sample_size_valid] = BasePBGKEModel(
                self.score_func, self.args.batch_size_eval,
                self.args.neg_sample_size_valid,
                hidden_dim, n_entities)
        if args.test:
            self.test_basic_models[self.args.neg_sample_size_test] = BasePBGKEModel(
                self.score_func, self.args.batch_size_eval,
                self.args.neg_sample_size_test,
                hidden_dim, n_entities)
        self.reset_parameters()

    def share_memory(self):
        # TODO(zhengda) we should make it work for parameters in score func
        self.entity_emb.share_memory()
        self.relation_emb.share_memory()

    def save_emb(self, path, dataset):
        self.entity_emb.save(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.save(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.save(path, dataset)

    def load_emb(self, path, dataset):
        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity.npy')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation.npy')
        self.score_func.load(path, dataset)

    def reset_parameters(self):
        self.entity_emb.init(self.emb_init)
        self.relation_emb.init(self.emb_init)
        self.score_func.reset_parameters()
        for param in self.parameters():
            if param.requires_grad:
                param.share_memory_()

    def forward_test(self, pos_g, neg_g, neg_head, neg_sample_size, logs, gpu_id=-1):
        with th.no_grad():
            pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, False)
            pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, False)

            batch_size = pos_g.number_of_edges()
            pos_scores = self.test_basic_models[neg_sample_size].predict_score(pos_g)
            pos_scores = F.logsigmoid(pos_scores).view(batch_size, -1)

            neg_scores = self.test_basic_models[neg_sample_size].predict_neg_score(
                pos_g, neg_g, self.entity_emb, neg_head, gpu_id=gpu_id, trace=False)
            neg_scores = F.logsigmoid(neg_scores).view(batch_size, -1)

            # We need to filter the positive edges in the negative graph.
            filter_bias = neg_g.edata['bias'].view(batch_size, -1)
            if self.args.gpu >= 0:
                filter_bias = filter_bias.cuda(self.args.gpu)
            neg_scores += filter_bias
            # To compute the rank of a positive edge among all negative edges,
            # we need to know how many negative edges have higher scores than
            # the positive edge.
            rankings = th.sum(neg_scores > pos_scores, dim=1) + 1

            for i in range(batch_size):
                ranking = rankings[i].item()
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

    # @profile
    def forward(self, pos_g, neg_g, neg_head, gpu_id=-1):
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)

        pos_score = self.train_basic_model.predict_score(pos_g)
        pos_score = F.logsigmoid(pos_score)
        neg_score = self.train_basic_model.predict_neg_score(pos_g, neg_g, self.entity_emb, neg_head,
                                                             gpu_id=gpu_id, trace=True)

        neg_score = neg_score.view(-1, self.args.neg_sample_size)
        # Adversarial sampling
        if self.args.neg_adversarial_sampling:
            neg_score = (F.softmax(neg_score * self.args.adversarial_temperature, dim=1).detach()
                         * F.logsigmoid(-neg_score)).sum(dim=1)
        else:
            neg_score = F.logsigmoid(-neg_score).mean(dim=1)

        # subsampling weight
        # TODO: add subsampling to new sampler
        if self.args.non_uni_weight:
            subsampling_weight = pos_g.edata['weight']
            pos_score = (pos_score * subsampling_weight).sum() / subsampling_weight.sum()
            neg_score = (neg_score * subsampling_weight).sum() / subsampling_weight.sum()
        else:
            pos_score = pos_score.mean()
            neg_score = neg_score.mean()

        # compute loss
        loss = -(pos_score + neg_score) / 2

        log = {'pos_loss': - pos_score.detach().item(),
               'neg_loss': - neg_score.detach().item(),
               'loss': loss.detach().item()}

        # regularization: TODO(zihao)
        #TODO: only reg ent&rel embeddings. other params to be added.
        if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
            coef, nm = self.args.regularization_coef, self.args.regularization_norm
            reg = coef * (self.entity_emb.curr_emb().norm(p=nm)**nm + self.relation_emb.curr_emb().norm(p=nm)**nm)
            log['regularization'] = reg.detach().item()
            loss += reg

        return loss, log

    def update(self):
        self.entity_emb.update()
        self.relation_emb.update()
