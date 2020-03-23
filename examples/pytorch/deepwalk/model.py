import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, device, args):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.device = device
        self.mixed_train = mixed_train
        self.neg_weight = neg_weight
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01).to(device))
        self.lookup_table[0] = 0.
        self.lookup_table[-1] = 1.

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def fast_learn_multi(self, pos_u, pos_v, neg_u, neg_v, lr):
        """ multi-sequence learning """
        # pos_u [batch_size, num_pos]
        # pos_v [batch_size, num_pos]
        # neg_u [batch_size, walk_length]
        # neg_v [batch_size, negative]
        # [batch_size, num_pos, dim]
        emb_u = self.u_embeddings.weight[pos_u]
        # [batch_size, num_pos, dim]
        emb_v = self.v_embeddings.weight[pos_v]
        # [batch_size, walk_length, dim]
        emb_neg_u = self.u_embeddings.weight[neg_u]
        # [batch_size, negative, dim]
        emb_neg_v = self.v_embeddings.weight[neg_v]

        if self.mixed_train:
            emb_u = emb_u.to(self.device)
            emb_v = emb_v.to(self.device)
            emb_neg_u = emb_neg_u.to(self.device)
            emb_neg_v = emb_neg_v.to(self.device)

        pos_score = torch.sum(torch.mul(emb_u, emb_v), dim=2)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        idx = torch.floor((pos_score + 6.01) / 0.01)
        idx = idx.long()
        # [batch_size, num_pos]
        sigmoid_score = self.lookup_table[idx]
        # [batch_size, num_pos, 1]
        sigmoid_score = (1 - sigmoid_score).unsqueeze(2)

        grad_v = sigmoid_score * emb_u
        grad_v = grad_v.view(-1, self.emb_dimension)
        grad_u = sigmoid_score * emb_v
        grad_u = grad_u.view(-1, self.emb_dimension)

        # [batch_size, walk_length, negative]
        neg_score = emb_neg_u.bmm(emb_neg_v.transpose(1,2)) 
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        idx = torch.floor((neg_score + 6.01) / 0.01)
        idx = idx.long()
        sigmoid_score = self.lookup_table[idx]
        sigmoid_score = - sigmoid_score

        # [batch_size, negative, dim]
        grad_neg_v = sigmoid_score.transpose(1,2).bmm(emb_neg_u)
        grad_neg_v = grad_neg_v.view(-1, self.emb_dimension)
        # [batch_size, walk_length, dim]
        grad_neg_u = sigmoid_score.bmm(emb_neg_v)
        grad_neg_u = grad_neg_u.view(-1, self.emb_dimension)

        grad_v *= lr
        grad_u *= lr
        grad_neg_v *= self.neg_weight * lr
        grad_neg_u *= self.neg_weight * lr 

        if self.mixed_train:
            grad_v = grad_v.cpu()
            grad_u = grad_u.cpu()
            grad_neg_v = grad_neg_v.cpu()
            grad_neg_u = grad_neg_u.cpu()
            pos_v = pos_v.cpu()
            pos_u = pos_u.cpu()
            neg_v = neg_v.cpu()
            neg_u = neg_u.cpu()

        self.v_embeddings.weight.index_add_(0, pos_v.view(-1), grad_v)
        self.v_embeddings.weight.index_add_(0, neg_v.view(-1), grad_neg_v)
        self.u_embeddings.weight.index_add_(0, pos_u.view(-1), grad_u)
        self.u_embeddings.weight.index_add_(0, neg_u.view(-1), grad_neg_u)

        return

    def fast_learn(self, pos_u, pos_v, neg_u, neg_v, lr):
        """ single-sequence learning """
        # pos_u [num_pos]
        # pos_v [num_pos]
        # neg_u [walk_length]
        # neg_v [negative]
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_u = self.u_embeddings(neg_u)
        emb_neg_v = self.v_embeddings(neg_v)

        pos_score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        idx = torch.floor((pos_score + 6.01) / 0.01)
        idx = idx.long()
        sigmoid_score = self.lookup_table[idx]
        sigmoid_score = (1 - sigmoid_score).unsqueeze(1)

        grad_v = torch.clone(sigmoid_score * self.u_embeddings.weight[pos_u])
        grad_u = torch.clone(sigmoid_score * self.v_embeddings.weight[pos_v])

        neg_score = emb_neg_u.mm(emb_neg_v.T) # [batch_size, negative_size]
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        idx = torch.floor((neg_score + 6.01) / 0.01)
        idx = idx.long()
        sigmoid_score = self.lookup_table[idx]
        sigmoid_score = - sigmoid_score

        #neg_size = neg_score.shape[1]
        grad_neg_v = torch.clone(sigmoid_score.T.mm(emb_neg_u))
        grad_neg_u = torch.clone(sigmoid_score.mm(emb_neg_v)) 


        self.v_embeddings.weight.index_add_(0, pos_v, lr * grad_v)
        self.v_embeddings.weight.index_add_(0, neg_v, lr * grad_neg_v)
        self.u_embeddings.weight.index_add_(0, pos_u, lr * grad_u)
        self.u_embeddings.weight.index_add_(0, neg_u, lr * grad_neg_u)

        return

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=6, min=-6)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        #return torch.mean(score + neg_score)
        return torch.sum(score), torch.sum(neg_score)

    def save_embedding(self, dataset, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (self.emb_size, self.emb_dimension))
            for wid in range(self.emb_size):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (str(dataset.id2node[wid]), e))
