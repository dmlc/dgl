import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import numpy as np

def init_emb2pos_index(trainer, args, batch_size):
    '''index version
    Usage:
        # emb_u.shape: [batch_size * walk_length, dim]
        batch_emb2posu = torch.index_select(emb_u, 0, pos_u_index)
    '''
    idx_list_u = []
    idx_list_v = []
    for b in range(batch_size):
        for i in range(args.walk_length):
            for j in range(i-args.window_size, i):
                if j >= 0:
                    idx_list_u.append(j + b * args.walk_length)
                    idx_list_v.append(i + b * args.walk_length)
            for j in range(i+1, i+1+args.window_size):
                if j < args.walk_length:
                    idx_list_u.append(j + b * args.walk_length)
                    idx_list_v.append(i + b * args.walk_length)

    # [num_pos * batch_size]
    index_emb_posu = torch.LongTensor(idx_list_u)
    index_emb_posv = torch.LongTensor(idx_list_v)

    return index_emb_posu, index_emb_posv

def init_emb2neg_index(trainer, args, batch_size):
    '''index version, emb_negv serves for fast negative sampling'''
    idx_list_u = []
    for b in range(batch_size):
        for i in range(args.walk_length):
            for j in range(i-args.window_size, i):
                if j >= 0:
                    idx_list_u += [i + b * args.walk_length] * args.negative
            for j in range(i+1, i+1+args.window_size):
                if j < args.walk_length:
                    idx_list_u += [i + b * args.walk_length] * args.negative
    
    idx_list_v = list(range(batch_size * args.walk_length)) * args.negative * args.window_size * 2
    random.shuffle(idx_list_v)
    idx_list_v = idx_list_v[:len(idx_list_u)]

    # [bs * walk_length * negative]
    index_emb_negu = torch.LongTensor(idx_list_u)
    index_emb_negv = torch.LongTensor(idx_list_v)

    return index_emb_negu, index_emb_negv

def init_grad_avg(trainer, args, batch_size):
    '''index version
    Usage:
        # emb_u.shape: [batch_size * walk_length, dim]
        batch_emb2posu = torch.index_select(emb_u, 0, pos_u_index)
    '''
    grad_avg = []
    for b in range(batch_size):
        for i in range(args.walk_length):
            if i < args.window_size:
                grad_avg.append(1. / float(i+args.window_size))
            elif i >= args.walk_length - args.window_size:
                grad_avg.append(1. / float(args.walk_length - i - 1 + args.window_size))
            else:
                grad_avg.append(0.5 / args.window_size)

    # [num_pos * batch_size]
    return torch.Tensor(grad_avg).unsqueeze(1)

def init_empty_grad(trainer, args, batch_size):
    grad_u = torch.zeros((batch_size * args.walk_length, trainer.emb_dimension))
    grad_v = torch.zeros((batch_size * args.walk_length, trainer.emb_dimension))

    return grad_u, grad_v

def adam(grad, state_sum, nodes, lr, device, args):
    grad_sum = (grad * grad).mean(1)
    if not args.only_gpu:
        grad_sum = grad_sum.cpu()
    state_sum.index_add_(0, nodes, grad_sum) # cpu
    std = state_sum[nodes].to(device)  # gpu
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    grad = (lr * grad / std_values) # gpu

    return grad

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, args):
        """ initialize embedding on CPU 

        Paremeters
        ----------
        emb_size int : number of nodes
        emb_dimension int : embedding dimension
        args argparse.Parser : arguments
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.mixed_train = args.mix
        self.neg_weight = args.neg_weight
        self.negative = args.negative
        self.device = torch.device("cpu")
        self.args = args
        # content embedding
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        # context embedding
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        if self.args.adam:
            self.state_sum_u = torch.zeros(emb_size)
            self.state_sum_v = torch.zeros(emb_size)

        # The lookup table is used for fast sigmoid computing
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.
        self.lookup_table[-1] = 1.

        # indexes to select positive/negative node pairs from batch_walks
        self.index_emb_posu, self.index_emb_posv = init_emb2pos_index(self, args, args.batch_size)
        self.index_emb_negu, self.index_emb_negv = init_emb2neg_index(self, args, args.batch_size)
        # coefficients for averaging the gradients
        self.grad_avg = init_grad_avg(self, args, args.batch_size)
        # gradients of nodes in batch_walks
        self.grad_u, self.grad_v = init_empty_grad(self, args, args.batch_size)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def share_memory(self):
        """ share the parameters across subprocesses """
        self.u_embeddings.weight.share_memory_()
        self.v_embeddings.weight.share_memory_()
        if self.args.adam:
            self.state_sum_u.share_memory_()
            self.state_sum_v.share_memory_()

    def set_device(self, gpu_id):
        self.device = torch.device("cuda:%d" % gpu_id)
        print("The device is", self.device)
        self.lookup_table = self.lookup_table.to(self.device)
        self.index_emb_posu = self.index_emb_posu.to(self.device)
        self.index_emb_posv = self.index_emb_posv.to(self.device)
        self.index_emb_negu = self.index_emb_negu.to(self.device)
        self.index_emb_negv = self.index_emb_negv.to(self.device)
        self.grad_u = self.grad_u.to(self.device)
        self.grad_v = self.grad_v.to(self.device)
        self.grad_avg = self.grad_avg.to(self.device)

    def all_to_device(self, gpu_id):
        """ move all of the parameters to a single GPU """
        self.device = torch.device("cuda:%d" % gpu_id)
        self.set_device(gpu_id)
        self.u_embeddings = self.u_embeddings.cuda(gpu_id)
        self.v_embeddings = self.v_embeddings.cuda(gpu_id)
        if self.args.adam:
            self.state_sum_u = self.state_sum_u.to(self.device)
            self.state_sum_v = self.state_sum_v.to(self.device)

    def fast_learn_super(self, batch_walks, lr, neg_nodes=None):
        """ Fast learn a batch of random walks
        Parameters
        ----------
        batch_walks : a list of node sequnces
        lr : current learning rate
        neg_nodes : a long tensor of sampled true negative nodes. If neg_nodes is None,
            then do negative sampling randomly from the nodes in batch_walks as an alternative.

        Usage example
        -------------
        batch_walks = [torch.LongTensor([1,2,3,4]), 
                       torch.LongTensor([2,3,4,2])])
        lr = 0.01
        neg_nodes = []
        """
        if self.args.adam:
            lr = self.args.lr

        # [batch_size, walk_length]
        nodes = torch.stack(batch_walks)
        if self.args.only_gpu:
            nodes = nodes.to(self.device)
            if neg_nodes is not None:
                neg_nodes = neg_nodes.to(self.device)
        emb_u = self.u_embeddings.weight[nodes].view(-1, self.emb_dimension).to(self.device)
        emb_v = self.v_embeddings.weight[nodes].view(-1, self.emb_dimension).to(self.device)

        ## Postive
        bs = len(batch_walks)
        if bs < self.args.batch_size:
            index_emb_posu, index_emb_posv = init_emb2pos_index(self, self.args, bs)
            index_emb_posu = index_emb_posu.to(self.device)
            index_emb_posv = index_emb_posv.to(self.device)
        else:
            index_emb_posu = self.index_emb_posu
            index_emb_posv = self.index_emb_posv

        # num_pos: the number of positive node pairs generated by a single walk sequence
        # [batch_size * num_pos, dim]
        emb_pos_u = torch.index_select(emb_u, 0, index_emb_posu)
        emb_pos_v = torch.index_select(emb_v, 0, index_emb_posv)

        pos_score = torch.sum(torch.mul(emb_pos_u, emb_pos_v), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        
        idx = torch.floor((pos_score + 6.01) / 0.01).long()
        # [batch_size * num_pos]
        sigmoid_score = self.lookup_table[idx]
        # [batch_size * num_pos, 1]
        sigmoid_score = (1 - sigmoid_score).unsqueeze(1)

        # [batch_size * num_pos, dim]
        if self.args.lap_norm > 0:
            grad_u_pos = sigmoid_score * emb_pos_v + self.args.lap_norm * (emb_pos_v - emb_pos_u)
            grad_v_pos = sigmoid_score * emb_pos_u + self.args.lap_norm * (emb_pos_u - emb_pos_v)
        else:
            grad_u_pos = sigmoid_score * emb_pos_v
            grad_v_pos = sigmoid_score * emb_pos_u
        # [batch_size * walk_length, dim]
        if bs < self.args.batch_size:
            grad_u, grad_v = init_empty_grad(self, self.args, bs)
            grad_u = grad_u.to(self.device)
            grad_v = grad_v.to(self.device)
        else:
            self.grad_u = self.grad_u.to(self.device)
            self.grad_u.zero_()
            self.grad_v = self.grad_v.to(self.device)
            self.grad_v.zero_()
            grad_u = self.grad_u
            grad_v = self.grad_v
        grad_u.index_add_(0, index_emb_posu, grad_u_pos)
        grad_v.index_add_(0, index_emb_posv, grad_v_pos)

        ## Negative
        if bs < self.args.batch_size:
            index_emb_negu, index_emb_negv = init_emb2neg_index(self, self.args, bs)
            index_emb_negu = index_emb_negu.to(self.device)
            index_emb_negv = index_emb_negv.to(self.device)
        else:
            index_emb_negu = self.index_emb_negu
            index_emb_negv = self.index_emb_negv

        emb_neg_u = torch.index_select(emb_u, 0, index_emb_negu)
        if neg_nodes is None:
            emb_neg_v = torch.index_select(emb_v, 0, index_emb_negv)
        else:
            emb_neg_v = self.v_embeddings.weight[neg_nodes].to(self.device).to(self.device)

        # [batch_size * walk_length * negative, dim]
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        idx = torch.floor((neg_score + 6.01) / 0.01).long()
        sigmoid_score = - self.lookup_table[idx]
        sigmoid_score = sigmoid_score.unsqueeze(1)

        grad_u_neg = self.args.neg_weight * sigmoid_score * emb_neg_v
        grad_v_neg = self.args.neg_weight * sigmoid_score * emb_neg_u

        grad_u.index_add_(0, index_emb_negu, grad_u_neg)
        if neg_nodes is None:
            grad_v.index_add_(0, index_emb_negv, grad_v_neg)

        ## Update
        nodes = nodes.view(-1)
        if self.args.avg_sgd:
            if bs < self.args.batch_size:
                grad_avg = init_grad_avg(self, self.args, bs).to(self.device)
            else:
                grad_avg = self.grad_avg
            grad_u = grad_avg * grad_u * lr
            grad_v = grad_avg * grad_v * lr
        elif self.args.sgd:
            grad_u = grad_u * lr
            grad_v = grad_v * lr
        elif self.args.adam:
            grad_u = adam(grad_u, self.state_sum_u, nodes, lr, self.device, self.args)
            grad_v = adam(grad_v, self.state_sum_v, nodes, lr, self.device, self.args)

        if self.mixed_train:
            grad_u = grad_u.cpu()
            grad_v = grad_v.cpu()
            if neg_nodes is not None:
                grad_v_neg = grad_v_neg.cpu()
        
        self.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
        self.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)            
        if neg_nodes is not None:
            self.v_embeddings.weight.data.index_add_(0, neg_nodes.view(-1), lr * grad_v_neg)
        return

    def fast_learn_multi(self, pos_u, pos_v, neg_u, neg_v, lr):
        """ multi-sequence learning, unused """
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

    def fast_learn_single(self, pos_u, pos_v, neg_u, neg_v, lr):
        """ single-sequence learning, unused """
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
        ''' unused '''
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
