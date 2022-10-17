import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Queue
from torch.nn import init


def init_emb2pos_index(walk_length, window_size, batch_size):
    """select embedding of positive nodes from a batch of node embeddings

    Return
    ------
    index_emb_posu torch.LongTensor : the indices of u_embeddings
    index_emb_posv torch.LongTensor : the indices of v_embeddings

    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2posu = torch.index_select(emb_u, 0, index_emb_posu)
    """
    idx_list_u = []
    idx_list_v = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
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
    """select embedding of negative nodes from a batch of node embeddings
    for fast negative sampling

    Return
    ------
    index_emb_negu torch.LongTensor : the indices of u_embeddings
    index_emb_negv torch.LongTensor : the indices of v_embeddings

    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2negu = torch.index_select(emb_u, 0, index_emb_negu)
    """
    idx_list_u = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    idx_list_u += [i + b * walk_length] * negative
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    idx_list_u += [i + b * walk_length] * negative

    idx_list_v = (
        list(range(batch_size * walk_length)) * negative * window_size * 2
    )
    random.shuffle(idx_list_v)
    idx_list_v = idx_list_v[: len(idx_list_u)]

    # [bs * walk_length * negative]
    index_emb_negu = torch.LongTensor(idx_list_u)
    index_emb_negv = torch.LongTensor(idx_list_v)

    return index_emb_negu, index_emb_negv


def init_weight(walk_length, window_size, batch_size):
    """init context weight"""
    weight = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    weight.append(1.0 - float(i - j - 1) / float(window_size))
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    weight.append(1.0 - float(j - i - 1) / float(window_size))

    # [num_pos * batch_size]
    return torch.Tensor(weight).unsqueeze(1)


def init_empty_grad(emb_dimension, walk_length, batch_size):
    """initialize gradient matrix"""
    grad_u = torch.zeros((batch_size * walk_length, emb_dimension))
    grad_v = torch.zeros((batch_size * walk_length, emb_dimension))

    return grad_u, grad_v


def adam(grad, state_sum, nodes, lr, device, only_gpu):
    """calculate gradients according to adam"""
    grad_sum = (grad * grad).mean(1)
    if not only_gpu:
        grad_sum = grad_sum.cpu()
    state_sum.index_add_(0, nodes, grad_sum)  # cpu
    std = state_sum[nodes].to(device)  # gpu
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    grad = lr * grad / std_values  # gpu

    return grad


def async_update(num_threads, model, queue):
    """asynchronous embedding update"""
    torch.set_num_threads(num_threads)
    while True:
        (grad_u, grad_v, grad_v_neg, nodes, neg_nodes) = queue.get()
        if grad_u is None:
            return
        with torch.no_grad():
            model.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
            model.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)
            if neg_nodes is not None:
                model.v_embeddings.weight.data.index_add_(
                    0, neg_nodes.view(-1), grad_v_neg
                )


class SkipGramModel(nn.Module):
    """Negative sampling based skip-gram"""

    def __init__(
        self,
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
        fast_neg,
        record_loss,
        norm,
        use_context_weight,
        async_update,
        num_threads,
    ):
        """initialize embedding on CPU

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
        fast_neg bool : do negative sampling inside a batch
        record_loss bool : print the loss during training
        norm bool : do normalizatin on the embedding after training
        use_context_weight : give different weights to the nodes in a context window
        async_update : asynchronous training
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
        self.fast_neg = fast_neg
        self.record_loss = record_loss
        self.norm = norm
        self.use_context_weight = use_context_weight
        self.async_update = async_update
        self.num_threads = num_threads

        # initialize the device as cpu
        self.device = torch.device("cpu")

        # content embedding
        self.u_embeddings = nn.Embedding(
            self.emb_size, self.emb_dimension, sparse=True
        )
        # context embedding
        self.v_embeddings = nn.Embedding(
            self.emb_size, self.emb_dimension, sparse=True
        )
        # initialze embedding
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

        # lookup_table is used for fast sigmoid computing
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.0
        self.lookup_table[-1] = 1.0
        if self.record_loss:
            self.logsigmoid_table = torch.log(
                torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
            )
            self.loss = []

        # indexes to select positive/negative node pairs from batch_walks
        self.index_emb_posu, self.index_emb_posv = init_emb2pos_index(
            self.walk_length, self.window_size, self.batch_size
        )
        self.index_emb_negu, self.index_emb_negv = init_emb2neg_index(
            self.walk_length, self.window_size, self.negative, self.batch_size
        )

        if self.use_context_weight:
            self.context_weight = init_weight(
                self.walk_length, self.window_size, self.batch_size
            )

        # adam
        self.state_sum_u = torch.zeros(self.emb_size)
        self.state_sum_v = torch.zeros(self.emb_size)

        # gradients of nodes in batch_walks
        self.grad_u, self.grad_v = init_empty_grad(
            self.emb_dimension, self.walk_length, self.batch_size
        )

    def create_async_update(self):
        """Set up the async update subprocess."""
        self.async_q = Queue(1)
        self.async_p = mp.Process(
            target=async_update, args=(self.num_threads, self, self.async_q)
        )
        self.async_p.start()

    def finish_async_update(self):
        """Notify the async update subprocess to quit."""
        self.async_q.put((None, None, None, None, None))
        self.async_p.join()

    def share_memory(self):
        """share the parameters across subprocesses"""
        self.u_embeddings.weight.share_memory_()
        self.v_embeddings.weight.share_memory_()
        self.state_sum_u.share_memory_()
        self.state_sum_v.share_memory_()

    def set_device(self, gpu_id):
        """set gpu device"""
        self.device = torch.device("cuda:%d" % gpu_id)
        print("The device is", self.device)
        self.lookup_table = self.lookup_table.to(self.device)
        if self.record_loss:
            self.logsigmoid_table = self.logsigmoid_table.to(self.device)
        self.index_emb_posu = self.index_emb_posu.to(self.device)
        self.index_emb_posv = self.index_emb_posv.to(self.device)
        self.index_emb_negu = self.index_emb_negu.to(self.device)
        self.index_emb_negv = self.index_emb_negv.to(self.device)
        self.grad_u = self.grad_u.to(self.device)
        self.grad_v = self.grad_v.to(self.device)
        if self.use_context_weight:
            self.context_weight = self.context_weight.to(self.device)

    def all_to_device(self, gpu_id):
        """move all of the parameters to a single GPU"""
        self.device = torch.device("cuda:%d" % gpu_id)
        self.set_device(gpu_id)
        self.u_embeddings = self.u_embeddings.cuda(gpu_id)
        self.v_embeddings = self.v_embeddings.cuda(gpu_id)
        self.state_sum_u = self.state_sum_u.to(self.device)
        self.state_sum_v = self.state_sum_v.to(self.device)

    def fast_sigmoid(self, score):
        """do fast sigmoid by looking up in a pre-defined table"""
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def fast_logsigmoid(self, score):
        """do fast logsigmoid by looking up in a pre-defined table"""
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.logsigmoid_table[idx]

    def fast_learn(self, batch_walks, neg_nodes=None):
        """Learn a batch of random walks in a fast way. It has the following features:
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
        emb_u = (
            self.u_embeddings(nodes)
            .view(-1, self.emb_dimension)
            .to(self.device)
        )
        emb_v = (
            self.v_embeddings(nodes)
            .view(-1, self.emb_dimension)
            .to(self.device)
        )

        ## Postive
        bs = len(batch_walks)
        if bs < self.batch_size:
            index_emb_posu, index_emb_posv = init_emb2pos_index(
                self.walk_length, self.window_size, bs
            )
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
        if self.record_loss:
            self.loss.append(torch.mean(self.fast_logsigmoid(pos_score)).item())

        # [batch_size * num_pos, dim]
        if self.lap_norm > 0:
            grad_u_pos = score * emb_pos_v + self.lap_norm * (
                emb_pos_v - emb_pos_u
            )
            grad_v_pos = score * emb_pos_u + self.lap_norm * (
                emb_pos_u - emb_pos_v
            )
        else:
            grad_u_pos = score * emb_pos_v
            grad_v_pos = score * emb_pos_u

        if self.use_context_weight:
            if bs < self.batch_size:
                context_weight = init_weight(
                    self.walk_length, self.window_size, bs
                ).to(self.device)
            else:
                context_weight = self.context_weight
            grad_u_pos *= context_weight
            grad_v_pos *= context_weight

        # [batch_size * walk_length, dim]
        if bs < self.batch_size:
            grad_u, grad_v = init_empty_grad(
                self.emb_dimension, self.walk_length, bs
            )
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
                self.walk_length, self.window_size, self.negative, bs
            )
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
        score = -self.fast_sigmoid(neg_score).unsqueeze(1)
        if self.record_loss:
            self.loss.append(
                self.negative
                * self.neg_weight
                * torch.mean(self.fast_logsigmoid(-neg_score)).item()
            )

        grad_u_neg = self.neg_weight * score * emb_neg_v
        grad_v_neg = self.neg_weight * score * emb_neg_u

        grad_u.index_add_(0, index_emb_negu, grad_u_neg)
        if neg_nodes is None:
            grad_v.index_add_(0, index_emb_negv, grad_v_neg)

        ## Update
        nodes = nodes.view(-1)

        # use adam optimizer
        grad_u = adam(
            grad_u, self.state_sum_u, nodes, lr, self.device, self.only_gpu
        )
        grad_v = adam(
            grad_v, self.state_sum_v, nodes, lr, self.device, self.only_gpu
        )
        if neg_nodes is not None:
            grad_v_neg = adam(
                grad_v_neg,
                self.state_sum_v,
                neg_nodes,
                lr,
                self.device,
                self.only_gpu,
            )

        if self.mixed_train:
            grad_u = grad_u.cpu()
            grad_v = grad_v.cpu()
            if neg_nodes is not None:
                grad_v_neg = grad_v_neg.cpu()
            else:
                grad_v_neg = None

            if self.async_update:
                grad_u.share_memory_()
                grad_v.share_memory_()
                nodes.share_memory_()
                if neg_nodes is not None:
                    neg_nodes.share_memory_()
                    grad_v_neg.share_memory_()
                self.async_q.put((grad_u, grad_v, grad_v_neg, nodes, neg_nodes))

        if not self.async_update:
            self.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
            self.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)
            if neg_nodes is not None:
                self.v_embeddings.weight.data.index_add_(
                    0, neg_nodes.view(-1), grad_v_neg
                )
        return

    def forward(self, pos_u, pos_v, neg_v):
        """Do forward and backward. It is designed for future use."""
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=6, min=-6)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # return torch.mean(score + neg_score)
        return torch.sum(score), torch.sum(neg_score)

    def save_embedding(self, dataset, file_name):
        """Write embedding to local file. Only used when node ids are numbers.

        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        if self.norm:
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(
                -1, 1
            )
        np.save(file_name, embedding)

    def save_embedding_pt(self, dataset, file_name):
        """For ogb leaderboard."""
        try:
            max_node_id = max(dataset.node2id.keys())
            if max_node_id + 1 != self.emb_size:
                print("WARNING: The node ids are not serial.")

            embedding = torch.zeros(max_node_id + 1, self.emb_dimension)
            index = torch.LongTensor(
                list(
                    map(
                        lambda id: dataset.id2node[id],
                        list(range(self.emb_size)),
                    )
                )
            )
            embedding.index_add_(0, index, self.u_embeddings.weight.cpu().data)

            if self.norm:
                embedding /= torch.sqrt(
                    torch.sum(embedding.mul(embedding), 1) + 1e-6
                ).unsqueeze(1)
            torch.save(embedding, file_name)
        except:
            self.save_embedding_pt_dgl_graph(dataset, file_name)

    def save_embedding_pt_dgl_graph(self, dataset, file_name):
        """For ogb leaderboard"""
        embedding = torch.zeros_like(self.u_embeddings.weight.cpu().data)
        valid_seeds = torch.LongTensor(dataset.valid_seeds)
        valid_embedding = self.u_embeddings.weight.cpu().data.index_select(
            0, valid_seeds
        )
        embedding.index_add_(0, valid_seeds, valid_embedding)

        if self.norm:
            embedding /= torch.sqrt(
                torch.sum(embedding.mul(embedding), 1) + 1e-6
            ).unsqueeze(1)

        torch.save(embedding, file_name)

    def save_embedding_txt(self, dataset, file_name):
        """Write embedding to local file. For future use.

        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        if self.norm:
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(
                -1, 1
            )
        with open(file_name, "w") as f:
            f.write("%d %d\n" % (self.emb_size, self.emb_dimension))
            for wid in range(self.emb_size):
                e = " ".join(map(lambda x: str(x), embedding[wid]))
                f.write("%s %s\n" % (str(dataset.id2node[wid]), e))
