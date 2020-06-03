import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import numpy as np

def init_emb2pos_index(walk_length, window_size, batch_size):
    ''' select embedding of positive nodes from a batch of node embeddings
    
    Return
    ------
    index_emb_posu torch.LongTensor : the indices of u_embeddings
    index_emb_posv torch.LongTensor : the indices of v_embeddings

    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2posu = torch.index_select(emb_u, 0, index_emb_posu)
    '''
    idx_list_u = []
    idx_list_v = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i-window_size, i):
                if j >= 0:
                    idx_list_u.append(j + b * walk_length)
                    idx_list_v.append(i + b * walk_length)
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    idx_list_u.append(j + b * walk_length)
                    idx_list_v.append(i + b * walk_length)

    # [num_pos * batch_size]
    index_emb_posu = torch.LongTensor(idx_list_u)
    index_emb_posv = torch.LongTensor(idx_list_v)

    return index_emb_posu, index_emb_posv

def init_emb2neg_index(walk_length, window_size, negative, batch_size):
    '''select embedding of negative nodes from a batch of node embeddings 
    for fast negative sampling
    
    Return
    ------
    index_emb_negu torch.LongTensor : the indices of u_embeddings
    index_emb_negv torch.LongTensor : the indices of v_embeddings

    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2negu = torch.index_select(emb_u, 0, index_emb_negu)
    '''
    idx_list_u = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i-window_size, i):
                if j >= 0:
                    idx_list_u += [i + b * walk_length] * negative
            for j in range(i+1, i+1+window_size):
                if j < walk_length:
                    idx_list_u += [i + b * walk_length] * negative
    
    idx_list_v = list(range(batch_size * walk_length))\
        * negative * window_size * 2
    random.shuffle(idx_list_v)
    idx_list_v = idx_list_v[:len(idx_list_u)]

    # [bs * walk_length * negative]
    index_emb_negu = torch.LongTensor(idx_list_u)
    index_emb_negv = torch.LongTensor(idx_list_v)

    return index_emb_negu, index_emb_negv

def init_grad_avg(walk_length, window_size, batch_size):
    '''select nodes' gradients from gradient matrix

    Usage
    -----

    '''
    grad_avg = []
    for b in range(batch_size):
        for i in range(walk_length):
            if i < window_size:
                grad_avg.append(1. / float(i+window_size))
            elif i >= walk_length - window_size:
                grad_avg.append(1. / float(walk_length - i - 1 + window_size))
            else:
                grad_avg.append(0.5 / window_size)

    # [num_pos * batch_size]
    return torch.Tensor(grad_avg).unsqueeze(1)

def init_empty_grad(emb_dimension, walk_length, batch_size):
    """ initialize gradient matrix """
    grad_u = torch.zeros((batch_size * walk_length, emb_dimension))
    grad_v = torch.zeros((batch_size * walk_length, emb_dimension))

    return grad_u, grad_v

def adam(grad, state_sum, nodes, lr, device, only_gpu):
    """ calculate gradients according to adam """
    grad_sum = (grad * grad).mean(1)
    if not only_gpu:
        grad_sum = grad_sum.cpu()
    state_sum.index_add_(0, nodes, grad_sum) # cpu
    std = state_sum[nodes].to(device)  # gpu
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    grad = (lr * grad / std_values) # gpu

    return grad

class SkipGramModel(nn.Module):
    """ Negative sampling based skip-gram """
    def __init__(self, 
        emb_size, 
        emb_dimension,
        walk_length,
        window_size,
        batch_size,
        only_cpu,
        only_gpu,
        mix,
        neg_weight,
        negative,
        lr,
        lap_norm,
        adam,
        sgd,
        avg_sgd,
        fast_neg,
        ):
        """ initialize embedding on CPU 

        Paremeters
        ----------
        emb_size int : number of nodes
        emb_dimension int : embedding dimension
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        batch_size int : number of node sequences in each batch
        only_cpu bool : training with CPU
        only_gpu bool : training with GPU
        mix bool : mixed training with CPU and GPU
        negative int : negative samples for each positve node pair
        neg_weight float : negative weight
        lr float : initial learning rate
        lap_norm float : weight of laplacian normalization
        adam bool : use adam for embedding updation
        sgd bool : use sgd for embedding updation
        avg_sgd bool : average gradients of sgd for embedding updation
        fast_neg bool : do negative sampling inside a batch
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.walk_length = walk_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.only_cpu = only_cpu
        self.only_gpu = only_gpu
        self.mixed_train = mix
        self.neg_weight = neg_weight
        self.negative = negative
        self.lr = lr
        self.lap_norm = lap_norm
        self.adam = adam
        self.sgd = sgd
        self.avg_sgd = avg_sgd
        self.fast_neg = fast_neg
        
        # initialize the device as cpu
        self.device = torch.device("cpu")

        # content embedding
        self.u_embeddings = nn.Embedding(
            self.emb_size, self.emb_dimension, sparse=True)
        # context embedding
        self.v_embeddings = nn.Embedding(
            self.emb_size, self.emb_dimension, sparse=True)
        # initialze embedding
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

        # lookup_table is used for fast sigmoid computing
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.
        self.lookup_table[-1] = 1.

        # indexes to select positive/negative node pairs from batch_walks
        self.index_emb_posu, self.index_emb_posv = init_emb2pos_index(
            self.walk_length,
            self.window_size,
            self.batch_size)
        if self.fast_neg:
            self.index_emb_negu, self.index_emb_negv = init_emb2neg_index(
                self.walk_length,
                self.window_size,
                self.negative,
                self.batch_size)

        # coefficients for averaging the gradients
        if self.avg_sgd:
            self.grad_avg = init_grad_avg(
                self.walk_length,
                self.window_size,
                self.batch_size)
        # adam
        if self.adam:
            self.state_sum_u = torch.zeros(self.emb_size)
            self.state_sum_v = torch.zeros(self.emb_size)

        # gradients of nodes in batch_walks
        self.grad_u, self.grad_v = init_empty_grad(
            self.emb_dimension,
            self.walk_length,
            self.batch_size)

    def share_memory(self):
        """ share the parameters across subprocesses """
        self.u_embeddings.weight.share_memory_()
        self.v_embeddings.weight.share_memory_()
        if self.adam:
            self.state_sum_u.share_memory_()
            self.state_sum_v.share_memory_()

    def set_device(self, gpu_id):
        """ set gpu device """
        self.device = torch.device("cuda:%d" % gpu_id)
        print("The device is", self.device)
        self.lookup_table = self.lookup_table.to(self.device)
        self.index_emb_posu = self.index_emb_posu.to(self.device)
        self.index_emb_posv = self.index_emb_posv.to(self.device)
        if self.fast_neg:
            self.index_emb_negu = self.index_emb_negu.to(self.device)
            self.index_emb_negv = self.index_emb_negv.to(self.device)
        self.grad_u = self.grad_u.to(self.device)
        self.grad_v = self.grad_v.to(self.device)
        if self.avg_sgd:
            self.grad_avg = self.grad_avg.to(self.device)

    def all_to_device(self, gpu_id):
        """ move all of the parameters to a single GPU """
        self.device = torch.device("cuda:%d" % gpu_id)
        self.set_device(gpu_id)
        self.u_embeddings = self.u_embeddings.cuda(gpu_id)
        self.v_embeddings = self.v_embeddings.cuda(gpu_id)
        if self.adam:
            self.state_sum_u = self.state_sum_u.to(self.device)
            self.state_sum_v = self.state_sum_v.to(self.device)

    def fast_sigmoid(self, score):
        """ do fast sigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def fast_learn(self, batch_walks, lr, neg_nodes=None):
        """ Learn a batch of random walks in a fast way. It has the following features:
            1. It calculating the gradients directly without the forward operation.
            2. It does sigmoid by a looking up table.

        Specifically, for each positive/negative node pair (i,j), the updating procedure is as following:
            score = self.fast_sigmoid(u_embedding[i].dot(v_embedding[j]))
            # label = 1 for positive samples; label = 0 for negative samples.
            u_embedding[i] += (label - score) * v_embedding[j]
            v_embedding[i] += (label - score) * u_embedding[j]

        Parameters
        ----------
        batch_walks list : a list of node sequnces
        lr float : current learning rate
        neg_nodes torch.LongTensor : a long tensor of sampled true negative nodes. If neg_nodes is None,
            then do negative sampling randomly from the nodes in batch_walks as an alternative.

        Usage example
        -------------
        batch_walks = [torch.LongTensor([1,2,3,4]), 
                       torch.LongTensor([2,3,4,2])])
        lr = 0.01
        neg_nodes = None
        """
        if self.adam:
            lr = self.lr

        # [batch_size, walk_length]
        if isinstance(batch_walks, list):
            nodes = torch.stack(batch_walks)
        elif isinstance(batch_walks, torch.LongTensor):
            nodes = batch_walks
        if self.only_gpu:
            nodes = nodes.to(self.device)
            if neg_nodes is not None:
                neg_nodes = neg_nodes.to(self.device)
        emb_u = self.u_embeddings(nodes).view(-1, self.emb_dimension).to(self.device)
        emb_v = self.v_embeddings(nodes).view(-1, self.emb_dimension).to(self.device)

        ## Postive
        bs = len(batch_walks)
        if bs < self.batch_size:
            index_emb_posu, index_emb_posv = init_emb2pos_index(
                self.walk_length, 
                self.window_size, 
                bs)
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
        # [batch_size * num_pos, 1]
        score = (1 - self.fast_sigmoid(pos_score)).unsqueeze(1)

        # [batch_size * num_pos, dim]
        if self.lap_norm > 0:
            grad_u_pos = score * emb_pos_v + self.lap_norm * (emb_pos_v - emb_pos_u)
            grad_v_pos = score * emb_pos_u + self.lap_norm * (emb_pos_u - emb_pos_v)
        else:
            grad_u_pos = score * emb_pos_v
            grad_v_pos = score * emb_pos_u
        # [batch_size * walk_length, dim]
        if bs < self.batch_size:
            grad_u, grad_v = init_empty_grad(
                self.emb_dimension, 
                self.walk_length, 
                bs)
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
        if bs < self.batch_size:
            index_emb_negu, index_emb_negv = init_emb2neg_index(
                self.walk_length, self.window_size, self.negative, bs)
            index_emb_negu = index_emb_negu.to(self.device)
            index_emb_negv = index_emb_negv.to(self.device)
        else:
            index_emb_negu = self.index_emb_negu
            index_emb_negv = self.index_emb_negv
        emb_neg_u = torch.index_select(emb_u, 0, index_emb_negu)
        
        if neg_nodes is None:
            emb_neg_v = torch.index_select(emb_v, 0, index_emb_negv)
        else:
            emb_neg_v = self.v_embeddings.weight[neg_nodes].to(self.device)

        # [batch_size * walk_length * negative, dim]
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        # [batch_size * walk_length * negative, 1]
        score = - self.fast_sigmoid(neg_score).unsqueeze(1)

        grad_u_neg = self.neg_weight * score * emb_neg_v
        grad_v_neg = self.neg_weight * score * emb_neg_u

        grad_u.index_add_(0, index_emb_negu, grad_u_neg)
        if neg_nodes is None:
            grad_v.index_add_(0, index_emb_negv, grad_v_neg)

        ## Update
        nodes = nodes.view(-1)
        if self.avg_sgd:
            # since the times that a node are performed backward propagation are different, 
            # we need to average the gradients by different weight.
            # e.g. for sequence [1, 2, 3, ...] with window_size = 5, we have positive node 
            # pairs [(1,2), (1, 3), (1,4), ...]. To average the gradients for each node, we
            # perform weighting on the gradients of node pairs. 
            # The weights are: [1/5, 1/5, ..., 1/6, ..., 1/10, ..., 1/6, ..., 1/5].
            if bs < self.batch_size:
                grad_avg = init_grad_avg(
                    self.walk_length,
                    self.window_size,
                    bs).to(self.device)
            else:
                grad_avg = self.grad_avg
            grad_u = grad_avg * grad_u * lr
            grad_v = grad_avg * grad_v * lr
        elif self.sgd:
            grad_u = grad_u * lr
            grad_v = grad_v * lr
        elif self.adam:
            # use adam optimizer
            grad_u = adam(grad_u, self.state_sum_u, nodes, lr, self.device, self.only_gpu)
            grad_v = adam(grad_v, self.state_sum_v, nodes, lr, self.device, self.only_gpu)

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

    def forward(self, pos_u, pos_v, neg_v):
        ''' Do forward and backward. It is designed for future use. '''
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
        """ Write embedding to local file.

        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        np.save(file_name, embedding)

    def save_embedding_txt(self, dataset, file_name):
        """ Write embedding to local file. For future use.

        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (self.emb_size, self.emb_dimension))
            for wid in range(self.emb_size):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (str(dataset.id2node[wid]), e))
