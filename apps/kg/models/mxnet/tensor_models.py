import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

from .score_fun import *
from .. import *

def logsigmoid(val):
    max_elem = nd.maximum(0., -val)
    z = nd.exp(-max_elem) + nd.exp(-val - max_elem)
    return -(max_elem + nd.log(z))

def to_device(val, gpu_id):
    return val.as_in_context(mx.gpu(gpu_id))

class ExternalEmbedding:
    def __init__(self, args, num, dim, ctx):
        self.gpu = args.gpu
        self.args = args
        self.trace = []

        self.emb = nd.empty((num, dim), dtype=np.float32, ctx=ctx)
        self.state_sum = nd.zeros((self.emb.shape[0]), dtype=np.float32, ctx=ctx)
        self.state_step = 0

    def init(self, emb_init):
        nd.random.uniform(-emb_init, emb_init,
                          shape=self.emb.shape, dtype=self.emb.dtype,
                          ctx=self.emb.context, out=self.emb)

    def share_memory(self):
        # TODO(zhengda) fix this later
        pass

    def __call__(self, idx, gpu_id=-1, trace=True):
        if self.emb.context != idx.context:
            idx = idx.as_in_context(self.emb.context)
        data = nd.take(self.emb, idx)
        if self.gpu >= 0:
            data = data.as_in_context(mx.gpu(self.gpu))
        data.attach_grad()
        if trace:
            self.trace.append((idx, data))
        return data

    def update(self):
        self.state_step += 1
        for idx, data in self.trace:
            grad = data.grad

            clr = self.args.lr
            #clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

            # the update is non-linear so indices must be unique
            grad_indices = idx
            grad_values = grad

            grad_sum = (grad_values * grad_values).mean(1)
            ctx = self.state_sum.context
            if ctx != grad_indices.context:
                grad_indices = grad_indices.as_in_context(ctx)
            if ctx != grad_sum.context:
                grad_sum = grad_sum.as_in_context(ctx)
            self.state_sum[grad_indices] += grad_sum
            std = self.state_sum[grad_indices]  # _sparse_mask
            std_values = nd.expand_dims(nd.sqrt(std) + 1e-10, 1)
            if self.gpu >= 0:
                std_values = std_values.as_in_context(mx.gpu(self.args.gpu))
            tmp = (-clr * grad_values / std_values)
            if tmp.context != ctx:
                tmp = tmp.as_in_context(ctx)
            # TODO(zhengda) the overhead is here.
            self.emb[grad_indices] = mx.nd.take(self.emb, grad_indices) + tmp
        self.trace = []

    def curr_emb(self):
        data = [data for _, data in self.trace]
        return nd.concat(*data, dim=0)

    def save(self, path, name):
        emb_fname = os.path.join(path, name+'.emb')
        state_fname = os.path.join(path, name+'.state')
        nd.save(emb_fname, self.emb)
        nd.save(state_fname, self.state_sum)

    def load(self, path, name):
        emb_fname = os.path.join(path, name+'.emb')
        state_fname = os.path.join(path, name+'.state')

        self.emb = nd.load(emb_fname)
        self.state_fname = nd.load(state_fname)

class PBGKEModel(gluon.Block):
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False):
        super(PBGKEModel, self).__init__()
        self.args = args
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = 2.0
        self.emb_init = (gamma + self.eps) / hidden_dim

        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        get_context = lambda gpu : mx.gpu(gpu) if gpu >= 0 else mx.cpu()
        self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim,
                                            mx.cpu() if args.mix_cpu_gpu else get_context(args.gpu))
        self.entity_emb.init(self.emb_init)
        # For RESCAL, relation_emb = relation_dim * entity_dim
        if model_name == 'RESCAL':
            rel_dim = relation_dim * entity_dim
        else:
            rel_dim = relation_dim
        self.relation_emb = ExternalEmbedding(args, n_relations, rel_dim, get_context(args.gpu))
        self.relation_emb.init(self.emb_init)

        if model_name == 'TransE':
            self.score_func = TransEScore(gamma)
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()

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

    def forward_test(self, pos_g, neg_g, neg_head, neg_sample_size, logs, gpu_id=-1):
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, False)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, False)

        batch_size = pos_g.number_of_edges()
        pos_scores = self.test_basic_models[neg_sample_size].predict_score(pos_g)
        pos_scores = logsigmoid(pos_scores).reshape(batch_size, -1)

        neg_scores = self.test_basic_models[neg_sample_size].predict_neg_score(
            pos_g, neg_g, self.entity_emb, neg_head, gpu_id=gpu_id, trace=False)
        neg_scores = logsigmoid(neg_scores).reshape(batch_size, -1)

        # We need to filter the positive edges in the negative graph.
        filter_bias = neg_g.edata['bias'].reshape(batch_size, -1)
        filter_bias = filter_bias.as_in_context(neg_scores.context)
        neg_scores += filter_bias
        # To compute the rank of a positive edge among all negative edges,
        # we need to know how many negative edges have higher scores than
        # the positive edge.
        rankings = nd.sum(neg_scores > pos_scores, axis=1) + 1
        rankings = rankings.asnumpy()
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
    def forward(self, pos_g, neg_g, neg_head, gpu_id=-1):
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)

        pos_score = self.train_basic_model.predict_score(pos_g)
        pos_score = logsigmoid(pos_score)
        if gpu_id >= 0:
            neg_score = self.train_basic_model.predict_neg_score(pos_g, neg_g, self.entity_emb, neg_head,
                                                                 to_device=to_device, gpu_id=gpu_id, trace=True)
        else:
            neg_score = self.train_basic_model.predict_neg_score(pos_g, neg_g, self.entity_emb, neg_head, trace=True)

        neg_score = neg_score.reshape(-1, self.args.neg_sample_size)
        # Adversarial sampling
        if self.args.neg_adversarial_sampling:
            neg_score = (nd.softmax(neg_score * self.args.adversarial_temperature, axis=1).detach()
                         * logsigmoid(-neg_score)).sum(axis=1)
        else:
            neg_score = logsigmoid(-neg_score).mean(axis=1)

        # subsampling weight
        # TODO: add subsampling to new sampler
        if self.args.non_uni_weight:
            subsampling_weight = pos_g.edata['weight']
            pos_score = (pos_score * subsampling_weight).sum() / subsampling_weight.sum()
            neg_score = (neg_score * subsampling_weight).sum() / subsampling_weight.sum()
        else:
            pos_score = pos_score.mean()
            neg_score = neg_score.mean()

        log = {'pos_loss': - pos_score.detach(),
               'neg_loss': - neg_score.detach()}

        norm = lambda x, p: nd.sum(nd.abs(x) ** p)
        # regularization: TODO(zihao)
        #TODO: only reg ent&rel embeddings. other params to be added.
        if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
            coef, nm = self.args.regularization_coef, self.args.regularization_norm
            reg = coef * (norm(self.entity_emb.curr_emb(), nm) + norm(self.relation_emb.curr_emb(), nm))
            log['regularization'] = reg.detach()
            loss = reg - (pos_score + neg_score) / 2
        else:
            # compute loss
            loss = -(pos_score + neg_score) / 2

        log['loss'] = loss.detach()
        return loss, log

    def share_memory(self):
        # TODO(zhengda) we should make it work for parameters in score func
        self.entity_emb.share_memory()
        self.relation_emb.share_memory()

    def save_emb(self, path, dataset):
        state_path = os.path.join(path, 'model.states')
        self.save_parameters(state_path)

        self.entity_emb.save(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.save(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.save(path, dataset)

    def load_emb(self, path, dataset):
        ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

        state_path = os.path.join(path, 'model.states')
        self.load_parameters(state_path)

        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset)

    def update(self):
        self.entity_emb.update()
        self.relation_emb.update()
