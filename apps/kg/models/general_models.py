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
        self.partition_book = None
        self.node_id = args.id

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
        elif model_name == 'TransR':
            projection_emb = ExternalEmbedding(args, n_relations, entity_dim * relation_dim,
                                               F.cpu() if args.mix_cpu_gpu else device)
            self.score_func = TransRScore(gamma, projection_emb, relation_dim, entity_dim)
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore(relation_dim, entity_dim)
            
        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)
        self.head_neg_prepare = self.score_func.create_neg_prepare(True)
        self.tail_neg_prepare = self.score_func.create_neg_prepare(False)

        self.reset_parameters()

    def set_partition_book(self, book):
        if book != None:
            self.partition_book = F.asnumpy(F.tensor(book))
        else:
            self.partition_book = None

    def share_memory(self):
        # TODO(zhengda) we should make it work for parameters in score func
        self.entity_emb.share_memory()
        self.relation_emb.share_memory()

    def save_emb(self, path, dataset):
        self.entity_emb.save(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.save(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.save(path, dataset+'_'+self.model_name)

    def load_emb(self, path, dataset):
        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset+'_'+self.model_name)

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

            neg_head, tail = self.head_neg_prepare(pos_g.edata['id'], num_chunks, neg_head, tail, gpu_id, trace)
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

            head, neg_tail = self.tail_neg_prepare(pos_g.edata['id'], num_chunks, head, neg_tail, gpu_id, trace)
            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)

        return neg_score

    def forward_test(self, pos_g, neg_g, logs, gpu_id=-1):
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, False)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, False)

        self.score_func.prepare(pos_g, gpu_id, False)

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

        self.score_func.prepare(pos_g, gpu_id, True)

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
        self.score_func.update()

    def pull_model(self, client, pos_g, neg_g):
        ######### TODO(chao): we still have room to improve the performance #########
        with th.no_grad():
            ######### pull entity id #########
            start = 0
            pull_count = 0
            entity_id = F.cat(seq=[pos_g.ndata['id'], neg_g.ndata['id']], dim=0)
            entity_id = np.unique(F.asnumpy(entity_id)) # remove the duplicated ID
            server_id = self.partition_book[entity_id]  # get the server-id mapping to data ID
            sorted_id = np.argsort(server_id)
            entity_id = entity_id[sorted_id] # sort data ID by server-id
            server, count = np.unique(server_id, return_counts=True) # get data size for each server
            for idx in range(len(server)):
                if server[idx] == self.node_id:
                    continue  # we don't need to pull the data on local machine
                client.pull(name='entity_emb', 
                    server_id=server[idx], 
                    id_tensor=F.tensor(entity_id[start:start+count[idx]]))
                start += count[idx]
                pull_count += 1
            ######### pull relation id #########
            relation_id = F.cat(seq=[pos_g.edata['id'], neg_g.edata['id']], dim=0)
            relation_id = np.unique(F.asnumpy(relation_id)) # remove the dupplicated ID
            # we pull relation_emb from server_0 by default
            client.pull(name='relation_emb', server_id=0, id_tensor=F.tensor(relation_id))
            pull_count += 1
            ######### wait pull result #########
            for idx in range(pull_count):
                msg = client.pull_wait()
                if msg.name == 'entity_emb':
                    self.entity_emb.emb[msg.id] = msg.data
                elif msg.anme == 'relation_emb':
                    self.relation_emb.emb[msg.id] = msg.data
                else:
                    raise RuntimeError('Unknown embedding name: %s' % msg.name)

    def dist_update(self, client):
        ######### TODO(chao): we still have room to improve the performance #########
        with th.no_grad():
            ######### update entity gradient #########
            start = 0
            for entity_id, entity_data in self.entity_emb.trace:
                entity_id = F.asnumpy(entity_id)
                grad_data = F.asnumpy(entity_data.grad.data)
                server_id = self.partition_book[entity_id] # get the server-id mapping to each server
                sorted_id = np.argsort(server_id)
                entity_id = entity_id[sorted_id] # sort data id
                grad_data = grad_data[sorted_id] # sort data gradient
                server, count = np.unique(server_id, return_counts=True) # get data size for each server
                entity_id = F.tensor(entity_id)
                grad_data = F.tensor(grad_data)
                grad_sum = (grad_data * grad_data).mean(1)
                for idx in range(len(server)):
                    end = start + count[idx]
                    if server[idx] == self.node_id: # update local model
                        self.entity_emb.partial_update(
                            grad_data[start:end], 
                            grad_sum[start:end],
                            entity_id[start:end])
                    else: # push gradient to remote machine
                        client.push(name='entity_emb_state',
                            id_tensor=entity_id[start:end],
                            data_tensor=grad_sum[start:end])
                        client.push(name='entity_emb', 
                            server_id=server[idx], 
                            id_tensor=entity_id[start:end], 
                            data_tensor=grad_data[start:end])
                    start += count[idx]
            ######### update relation gradient #########
            for relation_id, relation_data in self.relation_emb.trace:
                # we sync relation data on server_0 by default
                grad_data = relation_data.grad.data
                grad_sum = (grad_data * grad_data).mean(1)
                client.push(name='relation_emb_state',
                    server_id=0,
                    id_tensor=relation_id,
                    data_tensor=grad_sum)
                client.push(name='relation_emb', 
                    server_id=0, 
                    id_tensor=relation_id,
                    data_tensor=grad_data)