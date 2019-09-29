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
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT


class KEModel(nn.Module):
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False):
        super(KEModel, self).__init__()
        self.args = args
        self.model_name = model_name
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.hidden_dim = hidden_dim
        self.eps = 2.0
        self.gamma = gamma
        self.emb_init = (self.gamma + self.eps) / hidden_dim

        self.entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        self.relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        self.entity_emb = nn.Embedding(n_entities, self.entity_dim, sparse=args.sparse)
        # For RESCAL, relation_emb = relation_dim * entity_dim
        if model_name == 'RESCAL':
            rel_dim = self.relation_dim * self.entity_dim
        else:
            rel_dim = self.relation_dim
        self.relation_emb = nn.Embedding(n_relations, rel_dim)


        ### MODEL-SPECIFIC PARAMETERS ###
        if model_name == 'TransH':
            self.norm_vector = nn.Embedding(n_relations, self.relation_dim)
        elif model_name == 'TransR':
            self.transfer_matrix = nn.Embedding(n_relations, self.relation_dim * self.entity_dim)
        elif model_name == 'TransD':
            self.ent_transfer = nn.Embedding(n_entities, self.entity_dim)
            self.rel_transfer = nn.Embedding(n_relations, self.relation_dim)
        elif model_name == 'pRotatE':
            self.modulus = nn.Parameter(th.Tensor([[0.5 * self.emb_init]]))


        self.model_zoo = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'TransH': self.TransH,
            'TransR': self.TransR,
            'TransD': self.TransD,
            'RESCAL': self.RESCAL,
        }

        self.reset_parameters()

    def reset_parameters(self):
        INIT.uniform_(self.entity_emb.weight, -self.emb_init, self.emb_init)
        INIT.uniform_(self.relation_emb.weight, -self.emb_init, self.emb_init)
        if self.model_name == 'TransH':
            INIT.uniform_(self.norm_vector.weight, -self.emb_init, self.emb_init)
        elif self.model_name == 'TransR':
            INIT.uniform_(self.transfer_matrix.weight, -self.emb_init, self.emb_init)
        elif self.model_name == 'TransD':
            INIT.uniform_(self.ent_transfer.weight, -self.emb_init, self.emb_init)
            INIT.uniform_(self.rel_transfer.weight, -self.emb_init, self.emb_init)

        for param in self.parameters():
            if param.requires_grad:
                param.share_memory_()

    def TransE(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    # TODO: Test TransX and RESCAL (jin)
    # TODO: check other TransX
    def TransH(self, edges):
        def _transfer(e, norm):
            norm = F.normalize(norm, p=2, dim=-1)
            return e - th.sum(e * norm, -1, True) * norm

        rel = edges.data['emb']
        r_norm = edges.data['norm']
        head = _transfer(edges.src['emb'], r_norm)
        tail = _transfer(edges.dst['emb'], r_norm)
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def TransR(self, edges):
        def _transfer(transfer_matrix, embeddings):
            """
            :param transfer_matrix: shape: batch, d_rel, d_ent
            :param embeddings:  shape: batch, d_ent, 1
            :return: shape: batch, d_rel, 1
            """
            return th.matmul(transfer_matrix, embeddings)

        head = edges.src['emb'].unsqueeze(-1)
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel_transfer = edges.data['transfer_mat'].view(rel.shape[0], self.relation_dim, self.entity_dim)
        head = _transfer(rel_transfer, head).squeeze(-1)
        tail = _transfer(rel_transfer, tail).squeeze(-1)
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def TransD(self, edges):
        def _transfer(e, e_transfer, r_transfer):
            e = e + th.sum(e * e_transfer, -1, True) * r_transfer
            e_norm = F.normalize(e, p=2, dim=-1)
            return e_norm

        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        head_t = edges.src['entity_t']
        tail_t = edges.dst['entity_t']
        rel_t = edges.data['rel_t']
        head = _transfer(head, head_t, rel_t)
        tail = _transfer(tail, tail_t, rel_t)
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def DistMult(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def ComplEx(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, -1)}

    def RotatE(self, edges):
        pi = 3.14159265358979323846

        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        relation = edges.data['emb']

        phase_relation = relation / (self.emb_init / pi)

        real_rel, img_rel = th.cos(phase_relation), th.sin(phase_relation)

        real_score = real_head * real_rel - img_head * img_rel
        img_score = real_head * img_rel + img_head * real_rel
        real_score -= real_tail
        img_score -= img_tail

        score = th.stack([real_score, img_score], dim=0)
        score = th.norm(score, dim=0)
        score = self.gamma - score.sum(dim=-1)
        return {'score': score}

    def pRotatE(self, edges):
        pi = 3.14159262358979323846

        phase_head = edges.src['emb'] / (self.emb_init / pi)
        phase_tail = edges.dst['emb'] / (self.emb_init / pi)
        phase_rel = edges.data['emb'] / (self.emb_init / pi)

        score = (phase_head + phase_rel) - phase_tail
        score = score.sin().abs()
        score = (self.gamma - score.sum(dim=-1))
        score = score * self.modulus.item()
        return {'score': score}

    def RESCAL(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        score = head * th.matmul(rel, tail).squeeze(-1)
        # TODO: check if use self.gamma
        return {'score': th.sum(score, dim=-1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def score_func(self, edges):
        return self.model_zoo[self.model_name](edges)

    def forward_test(self, g, y, logs, mode=None):
        with th.no_grad():
            if self.args.cuda:
                y = y.cuda()

            batch_size = len(y)
            score = self.predict_score(g).view(batch_size, -1)
            filter_bias = g.edata['bias'].view(batch_size, -1)
            score += filter_bias
            argsort = th.argsort(score, dim=1, descending=True)

            for i in range(batch_size):
                ranking = (argsort[i, :] == y[i]).nonzero()
                assert ranking.size(0) == 1
                ranking = 1 + ranking.item()

                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })

    def forward(self, pos_g, neg_g, neg_head):
        if self.args.cuda:
            self.to_gpu(pos_g)
            self.to_gpu(neg_g)

        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'])
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'])

        neg_g.ndata['emb'] = self.entity_emb(neg_g.ndata['id'])
        neg_g.edata['emb'] = self.relation_emb(neg_g.edata['id'])

        # add other embeddings for TransH, TransR, TransD, RESCAL
        if self.model_name == 'TransH':
            pos_g.edata['norm'] = self.norm_vector(pos_g.edata['id'])
            neg_g.edata['norm'] = self.norm_vector(neg_g.edata['id'])
        elif self.model_name == 'TransR':
            pos_g.edata['transfer_mat'] = self.transfer_matrix(pos_g.edata['id'])
            neg_g.edata['transfer_mat'] = self.transfer_matrix(neg_g.edata['id'])
        elif self.model_name == 'TransD':
            pos_g.ndata['entity_t'] = self.ent_transfer(pos_g.ndata['id'])
            pos_g.edata['rel_t'] = self.rel_transfer(pos_g.edata['id'])
            neg_g.ndata['entity_t'] = self.ent_transfer(neg_g.ndata['id'])
            neg_g.edata['rel_t'] = self.rel_transfer(neg_g.edata['id'])
        elif self.model_name == 'RESCAL':
            pass

        pos_g.apply_edges(self.score_func)
        neg_g.apply_edges(self.score_func)

        # compute score
        pos_score = pos_g.edata['score']
        neg_score = neg_g.edata['score'].view(-1, self.args.neg_sample_size)
        pos_score = F.logsigmoid(pos_score)
        # Adversarial sampling
        if self.args.neg_adversarial_sampling: 
            neg_score = (F.softmax(neg_score * self.args.adversarial_temperature, dim=1).detach()
                         * F.logsigmoid(-neg_score)).sum(dim=1)
        else:
            neg_score = F.logsigmoid(-neg_score).mean(dim=1)

        # subsampling weight
        # TODO: add subsampling to new sampler (after that, set default value of args.uni_weight)
        if self.args.uni_weight:
            pos_score = pos_score.mean()
            neg_score = neg_score.mean()
        else:
            subsampling_weight = pos_g.edata['weight']
            pos_score = (pos_score * subsampling_weight).sum() / subsampling_weight.sum()
            neg_score = (neg_score * subsampling_weight).sum() / subsampling_weight.sum()

        # compute loss
        loss = -(pos_score + neg_score) / 2

        log = {'pos_loss': - pos_score.detach().item(),
               'neg_loss': - neg_score.detach().item(),
               'loss': loss.detach().item()}

        # regularization: TODO(zihao)
        #TODO: only reg ent&rel embeddings. other params to be added.
        if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
            coef, nm = self.args.regularization_coef, self.args.regularization_norm
            reg = coef * (self.entity_emb.weight.norm(p=nm)**nm + self.relation_emb.weight.norm(p=nm)**nm)
            log['regularization'] = reg
            loss += reg

        return loss, log

    def predict_score(self, pos_g):
        with th.no_grad():
            if self.args.cuda:
                self.to_gpu(pos_g)

            pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'])
            pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'])

            # add other embeddings for TransH, TransR, TransD, RESCAL
            if self.model_name == 'TransH':
                pos_g.edata['norm'] = self.norm_vector(pos_g.edata['id'])
            elif self.model_name == 'TransR':
                pos_g.edata['transfer_mat'] = self.transfer_matrix(pos_g.edata['id'])
            elif self.model_name == 'TransD':
                pos_g.ndata['entity_t'] = self.ent_transfer(pos_g.ndata['id'])
                pos_g.edata['rel_t'] = self.rel_transfer(pos_g.edata['id'])
            elif self.model_name == 'RESCAL':
                pass

        pos_g.apply_edges(self.score_func)
        pos_score = F.logsigmoid(pos_g.edata['score'])

        return pos_score

    def to_gpu(self, g):
        g.to(th.device('cuda:0'))
