import os
import numpy as np
import dgl.backend as F

backend = os.environ.get('DGLBACKEND')
if backend.lower() == 'mxnet':
    from .mxnet.tensor_models import logsigmoid
    from .mxnet.tensor_models import get_device
    from .mxnet.tensor_models import norm
    from .mxnet.tensor_models import get_scalar
    from .mxnet.tensor_models import reshape
    from .mxnet.tensor_models import cuda
    from .mxnet.tensor_models import ExternalEmbedding
    from .mxnet.score_fun import *
else:
    from .pytorch.tensor_models import logsigmoid
    from .pytorch.tensor_models import get_device
    from .pytorch.tensor_models import norm
    from .pytorch.tensor_models import get_scalar
    from .pytorch.tensor_models import reshape
    from .pytorch.tensor_models import cuda
    from .pytorch.tensor_models import ExternalEmbedding
    from .pytorch.score_fun import *

class KEModel(object):
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False):
        super(KEModel, self).__init__()
        self.args = args
        self.n_entities = n_entities
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = 2.0
        self.emb_init = (gamma + self.eps) / hidden_dim

        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        device = get_device(args)
        self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim,
                                            F.cpu() if args.mix_cpu_gpu else device)
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
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore(relation_dim, entity_dim)
            
        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)

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
        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset)

    def reset_parameters(self):
        self.entity_emb.init(self.emb_init)
        self.relation_emb.init(self.emb_init)
        self.score_func.reset_parameters()

    def predict_score(self, g):
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g, to_device=None, gpu_id=-1, trace=False):
        num_chunks = neg_g.num_chunks
        chunk_size = neg_g.chunk_size
        neg_sample_size = neg_g.neg_sample_size
        if neg_g.neg_head:
            neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
            neg_head = self.entity_emb(neg_head_ids, gpu_id, trace)
            _, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                tail_ids = to_device(tail_ids, gpu_id)
            tail = pos_g.ndata['emb'][tail_ids]
            rel = pos_g.edata['emb']
            neg_score = self.head_neg_score(neg_head, rel, tail,
                                            num_chunks, chunk_size, neg_sample_size)
        else:
            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            neg_tail = self.entity_emb(neg_tail_ids, gpu_id, trace)
            head_ids, _ = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                head_ids = to_device(head_ids, gpu_id)
            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']
            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)

        return neg_score

    def forward_test(self, pos_g, neg_g, logs, gpu_id=-1):
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, False)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, False)

        batch_size = pos_g.number_of_edges()
        pos_scores = self.predict_score(pos_g)
        pos_scores = reshape(logsigmoid(pos_scores), batch_size, -1)

        neg_scores = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                            gpu_id=gpu_id, trace=False)
        neg_scores = reshape(logsigmoid(neg_scores), batch_size, -1)

        # We need to filter the positive edges in the negative graph.
        filter_bias = reshape(neg_g.edata['bias'], batch_size, -1)
        if self.args.gpu >= 0:
            filter_bias = cuda(filter_bias, self.args.gpu)
        neg_scores += filter_bias
        # To compute the rank of a positive edge among all negative edges,
        # we need to know how many negative edges have higher scores than
        # the positive edge.
        rankings = F.sum(neg_scores > pos_scores, dim=1) + 1
        rankings = F.asnumpy(rankings)
        for i in range(batch_size):
            ranking = rankings[i]
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0
            })

    # @profile
    def forward(self, pos_g, neg_g, gpu_id=-1):
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)

        pos_score = self.predict_score(pos_g)
        pos_score = logsigmoid(pos_score)
        if gpu_id >= 0:
            neg_score = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                               gpu_id=gpu_id, trace=True)
        else:
            neg_score = self.predict_neg_score(pos_g, neg_g, trace=True)

        neg_score = reshape(neg_score, -1, neg_g.neg_sample_size)
        # Adversarial sampling
        if self.args.neg_adversarial_sampling:
            neg_score = F.sum(F.softmax(neg_score * self.args.adversarial_temperature, dim=1).detach()
                         * logsigmoid(-neg_score), dim=1)
        else:
            neg_score = F.mean(logsigmoid(-neg_score), dim=1)

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

        log = {'pos_loss': - get_scalar(pos_score),
               'neg_loss': - get_scalar(neg_score),
               'loss': get_scalar(loss)}

        # regularization: TODO(zihao)
        #TODO: only reg ent&rel embeddings. other params to be added.
        if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
            coef, nm = self.args.regularization_coef, self.args.regularization_norm
            reg = coef * (norm(self.entity_emb.curr_emb(), nm) + norm(self.relation_emb.curr_emb(), nm))
            log['regularization'] = get_scalar(reg)
            loss = loss + reg

        return loss, log

    def update(self):
        self.entity_emb.update()
        self.relation_emb.update()
