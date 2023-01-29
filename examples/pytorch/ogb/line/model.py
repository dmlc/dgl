import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Queue
from torch.nn import init


def init_emb2neg_index(negative, batch_size):
    """select embedding of negative nodes from a batch of node embeddings
    for fast negative sampling

    Return
    ------
    index_emb_negu torch.LongTensor : the indices of u_embeddings
    index_emb_negv torch.LongTensor : the indices of v_embeddings

    Usage
    -----
    # emb_u.shape: [batch_size, dim]
    batch_emb2negu = torch.index_select(emb_u, 0, index_emb_negu)
    """
    idx_list_u = list(range(batch_size)) * negative
    idx_list_v = list(range(batch_size)) * negative
    random.shuffle(idx_list_v)

    index_emb_negu = torch.LongTensor(idx_list_u)
    index_emb_negv = torch.LongTensor(idx_list_v)

    return index_emb_negu, index_emb_negv


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
    """Asynchronous embedding update for entity embeddings."""
    torch.set_num_threads(num_threads)
    print("async start")
    while True:
        (grad_u, grad_v, grad_v_neg, nodes, neg_nodes, first_flag) = queue.get()
        if grad_u is None:
            return
        with torch.no_grad():
            if first_flag:
                model.fst_u_embeddings.weight.data.index_add_(
                    0, nodes[:, 0], grad_u
                )
                model.fst_u_embeddings.weight.data.index_add_(
                    0, nodes[:, 1], grad_v
                )
                if neg_nodes is not None:
                    model.fst_u_embeddings.weight.data.index_add_(
                        0, neg_nodes, grad_v_neg
                    )
            else:
                model.snd_u_embeddings.weight.data.index_add_(
                    0, nodes[:, 0], grad_u
                )
                model.snd_v_embeddings.weight.data.index_add_(
                    0, nodes[:, 1], grad_v
                )
                if neg_nodes is not None:
                    model.snd_v_embeddings.weight.data.index_add_(
                        0, neg_nodes, grad_v_neg
                    )


class SkipGramModel(nn.Module):
    """Negative sampling based skip-gram"""

    def __init__(
        self,
        emb_size,
        emb_dimension,
        batch_size,
        only_cpu,
        only_gpu,
        only_fst,
        only_snd,
        mix,
        neg_weight,
        negative,
        lr,
        lap_norm,
        fast_neg,
        record_loss,
        async_update,
        num_threads,
    ):
        """initialize embedding on CPU

        Paremeters
        ----------
        emb_size int : number of nodes
        emb_dimension int : embedding dimension
        batch_size int : number of node sequences in each batch
        only_cpu bool : training with CPU
        only_gpu bool : training with GPU
        only_fst bool : only embedding for first-order proximity
        only_snd bool : only embedding for second-order proximity
        mix bool : mixed training with CPU and GPU
        negative int : negative samples for each positve node pair
        neg_weight float : negative weight
        lr float : initial learning rate
        lap_norm float : weight of laplacian normalization
        fast_neg bool : do negative sampling inside a batch
        record_loss bool : print the loss during training
        use_context_weight : give different weights to the nodes in a context window
        async_update : asynchronous training
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.only_cpu = only_cpu
        self.only_gpu = only_gpu
        if only_fst:
            self.fst = True
            self.snd = False
            self.emb_dimension = emb_dimension
        elif only_snd:
            self.fst = False
            self.snd = True
            self.emb_dimension = emb_dimension
        else:
            self.fst = True
            self.snd = True
            self.emb_dimension = int(emb_dimension / 2)
        self.mixed_train = mix
        self.neg_weight = neg_weight
        self.negative = negative
        self.lr = lr
        self.lap_norm = lap_norm
        self.fast_neg = fast_neg
        self.record_loss = record_loss
        self.async_update = async_update
        self.num_threads = num_threads

        # initialize the device as cpu
        self.device = torch.device("cpu")

        # embedding
        initrange = 1.0 / self.emb_dimension
        if self.fst:
            self.fst_u_embeddings = nn.Embedding(
                self.emb_size, self.emb_dimension, sparse=True
            )
            init.uniform_(
                self.fst_u_embeddings.weight.data, -initrange, initrange
            )
        if self.snd:
            self.snd_u_embeddings = nn.Embedding(
                self.emb_size, self.emb_dimension, sparse=True
            )
            init.uniform_(
                self.snd_u_embeddings.weight.data, -initrange, initrange
            )
            self.snd_v_embeddings = nn.Embedding(
                self.emb_size, self.emb_dimension, sparse=True
            )
            init.constant_(self.snd_v_embeddings.weight.data, 0)

        # lookup_table is used for fast sigmoid computing
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.0
        self.lookup_table[-1] = 1.0
        if self.record_loss:
            self.logsigmoid_table = torch.log(
                torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
            )
            self.loss_fst = []
            self.loss_snd = []

        # indexes to select positive/negative node pairs from batch_walks
        self.index_emb_negu, self.index_emb_negv = init_emb2neg_index(
            self.negative, self.batch_size
        )

        # adam
        if self.fst:
            self.fst_state_sum_u = torch.zeros(self.emb_size)
        if self.snd:
            self.snd_state_sum_u = torch.zeros(self.emb_size)
            self.snd_state_sum_v = torch.zeros(self.emb_size)

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
        if self.fst:
            self.fst_u_embeddings.weight.share_memory_()
            self.fst_state_sum_u.share_memory_()
        if self.snd:
            self.snd_u_embeddings.weight.share_memory_()
            self.snd_v_embeddings.weight.share_memory_()
            self.snd_state_sum_u.share_memory_()
            self.snd_state_sum_v.share_memory_()

    def set_device(self, gpu_id):
        """set gpu device"""
        self.device = torch.device("cuda:%d" % gpu_id)
        print("The device is", self.device)
        self.lookup_table = self.lookup_table.to(self.device)
        if self.record_loss:
            self.logsigmoid_table = self.logsigmoid_table.to(self.device)
        self.index_emb_negu = self.index_emb_negu.to(self.device)
        self.index_emb_negv = self.index_emb_negv.to(self.device)

    def all_to_device(self, gpu_id):
        """move all of the parameters to a single GPU"""
        self.device = torch.device("cuda:%d" % gpu_id)
        self.set_device(gpu_id)
        if self.fst:
            self.fst_u_embeddings = self.fst_u_embeddings.cuda(gpu_id)
            self.fst_state_sum_u = self.fst_state_sum_u.to(self.device)
        if self.snd:
            self.snd_u_embeddings = self.snd_u_embeddings.cuda(gpu_id)
            self.snd_v_embeddings = self.snd_v_embeddings.cuda(gpu_id)
            self.snd_state_sum_u = self.snd_state_sum_u.to(self.device)
            self.snd_state_sum_v = self.snd_state_sum_v.to(self.device)

    def fast_sigmoid(self, score):
        """do fast sigmoid by looking up in a pre-defined table"""
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def fast_logsigmoid(self, score):
        """do fast logsigmoid by looking up in a pre-defined table"""
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.logsigmoid_table[idx]

    def fast_pos_bp(self, emb_pos_u, emb_pos_v, first_flag):
        """get grad for positve samples"""
        pos_score = torch.sum(torch.mul(emb_pos_u, emb_pos_v), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        # [batch_size, 1]
        score = (1 - self.fast_sigmoid(pos_score)).unsqueeze(1)
        if self.record_loss:
            if first_flag:
                self.loss_fst.append(
                    torch.mean(self.fast_logsigmoid(pos_score)).item()
                )
            else:
                self.loss_snd.append(
                    torch.mean(self.fast_logsigmoid(pos_score)).item()
                )

        # [batch_size, dim]
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

        return grad_u_pos, grad_v_pos

    def fast_neg_bp(self, emb_neg_u, emb_neg_v, first_flag):
        """get grad for negative samples"""
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        # [batch_size * negative, 1]
        score = -self.fast_sigmoid(neg_score).unsqueeze(1)
        if self.record_loss:
            if first_flag:
                self.loss_fst.append(
                    self.negative
                    * self.neg_weight
                    * torch.mean(self.fast_logsigmoid(-neg_score)).item()
                )
            else:
                self.loss_snd.append(
                    self.negative
                    * self.neg_weight
                    * torch.mean(self.fast_logsigmoid(-neg_score)).item()
                )

        grad_u_neg = self.neg_weight * score * emb_neg_v
        grad_v_neg = self.neg_weight * score * emb_neg_u

        return grad_u_neg, grad_v_neg

    def fast_learn(self, batch_edges, neg_nodes=None):
        """Learn a batch of edges in a fast way. It has the following features:
            1. It calculating the gradients directly without the forward operation.
            2. It does sigmoid by a looking up table.

        Specifically, for each positive/negative node pair (i,j), the updating procedure is as following:
            score = self.fast_sigmoid(u_embedding[i].dot(v_embedding[j]))
            # label = 1 for positive samples; label = 0 for negative samples.
            u_embedding[i] += (label - score) * v_embedding[j]
            v_embedding[i] += (label - score) * u_embedding[j]

        Parameters
        ----------
        batch_edges list : a list of node sequnces
        neg_nodes torch.LongTensor : a long tensor of sampled true negative nodes. If neg_nodes is None,
            then do negative sampling randomly from the nodes in batch_walks as an alternative.

        Usage example
        -------------
        batch_walks = torch.LongTensor([[1,2], [3,4], [5,6]])
        neg_nodes = None
        """
        lr = self.lr

        # [batch_size, 2]
        nodes = batch_edges
        if self.only_gpu:
            nodes = nodes.to(self.device)
            if neg_nodes is not None:
                neg_nodes = neg_nodes.to(self.device)
        bs = len(nodes)

        if self.fst:
            emb_u = (
                self.fst_u_embeddings(nodes[:, 0])
                .view(-1, self.emb_dimension)
                .to(self.device)
            )
            emb_v = (
                self.fst_u_embeddings(nodes[:, 1])
                .view(-1, self.emb_dimension)
                .to(self.device)
            )

            ## Postive
            emb_pos_u, emb_pos_v = emb_u, emb_v
            grad_u_pos, grad_v_pos = self.fast_pos_bp(
                emb_pos_u, emb_pos_v, True
            )

            ## Negative
            emb_neg_u = emb_pos_u.repeat((self.negative, 1))

            if bs < self.batch_size:
                index_emb_negu, index_emb_negv = init_emb2neg_index(
                    self.negative, bs
                )
                index_emb_negu = index_emb_negu.to(self.device)
                index_emb_negv = index_emb_negv.to(self.device)
            else:
                index_emb_negu = self.index_emb_negu
                index_emb_negv = self.index_emb_negv

            if neg_nodes is None:
                emb_neg_v = torch.index_select(emb_v, 0, index_emb_negv)
            else:
                emb_neg_v = self.fst_u_embeddings.weight[neg_nodes].to(
                    self.device
                )

            grad_u_neg, grad_v_neg = self.fast_neg_bp(
                emb_neg_u, emb_neg_v, True
            )

            ## Update
            grad_u_pos.index_add_(0, index_emb_negu, grad_u_neg)
            grad_u = grad_u_pos
            if neg_nodes is None:
                grad_v_pos.index_add_(0, index_emb_negv, grad_v_neg)
                grad_v = grad_v_pos
            else:
                grad_v = grad_v_pos

            # use adam optimizer
            grad_u = adam(
                grad_u,
                self.fst_state_sum_u,
                nodes[:, 0],
                lr,
                self.device,
                self.only_gpu,
            )
            grad_v = adam(
                grad_v,
                self.fst_state_sum_u,
                nodes[:, 1],
                lr,
                self.device,
                self.only_gpu,
            )
            if neg_nodes is not None:
                grad_v_neg = adam(
                    grad_v_neg,
                    self.fst_state_sum_u,
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
                    self.async_q.put(
                        (grad_u, grad_v, grad_v_neg, nodes, neg_nodes, True)
                    )

            if not self.async_update:
                self.fst_u_embeddings.weight.data.index_add_(
                    0, nodes[:, 0], grad_u
                )
                self.fst_u_embeddings.weight.data.index_add_(
                    0, nodes[:, 1], grad_v
                )
                if neg_nodes is not None:
                    self.fst_u_embeddings.weight.data.index_add_(
                        0, neg_nodes, grad_v_neg
                    )

        if self.snd:
            emb_u = (
                self.snd_u_embeddings(nodes[:, 0])
                .view(-1, self.emb_dimension)
                .to(self.device)
            )
            emb_v = (
                self.snd_v_embeddings(nodes[:, 1])
                .view(-1, self.emb_dimension)
                .to(self.device)
            )

            ## Postive
            emb_pos_u, emb_pos_v = emb_u, emb_v
            grad_u_pos, grad_v_pos = self.fast_pos_bp(
                emb_pos_u, emb_pos_v, False
            )

            ## Negative
            emb_neg_u = emb_pos_u.repeat((self.negative, 1))

            if bs < self.batch_size:
                index_emb_negu, index_emb_negv = init_emb2neg_index(
                    self.negative, bs
                )
                index_emb_negu = index_emb_negu.to(self.device)
                index_emb_negv = index_emb_negv.to(self.device)
            else:
                index_emb_negu = self.index_emb_negu
                index_emb_negv = self.index_emb_negv

            if neg_nodes is None:
                emb_neg_v = torch.index_select(emb_v, 0, index_emb_negv)
            else:
                emb_neg_v = self.snd_v_embeddings.weight[neg_nodes].to(
                    self.device
                )

            grad_u_neg, grad_v_neg = self.fast_neg_bp(
                emb_neg_u, emb_neg_v, False
            )

            ## Update
            grad_u_pos.index_add_(0, index_emb_negu, grad_u_neg)
            grad_u = grad_u_pos
            if neg_nodes is None:
                grad_v_pos.index_add_(0, index_emb_negv, grad_v_neg)
                grad_v = grad_v_pos
            else:
                grad_v = grad_v_pos

            # use adam optimizer
            grad_u = adam(
                grad_u,
                self.snd_state_sum_u,
                nodes[:, 0],
                lr,
                self.device,
                self.only_gpu,
            )
            grad_v = adam(
                grad_v,
                self.snd_state_sum_v,
                nodes[:, 1],
                lr,
                self.device,
                self.only_gpu,
            )
            if neg_nodes is not None:
                grad_v_neg = adam(
                    grad_v_neg,
                    self.snd_state_sum_v,
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
                    self.async_q.put(
                        (grad_u, grad_v, grad_v_neg, nodes, neg_nodes, False)
                    )

            if not self.async_update:
                self.snd_u_embeddings.weight.data.index_add_(
                    0, nodes[:, 0], grad_u
                )
                self.snd_v_embeddings.weight.data.index_add_(
                    0, nodes[:, 1], grad_v
                )
                if neg_nodes is not None:
                    self.snd_v_embeddings.weight.data.index_add_(
                        0, neg_nodes, grad_v_neg
                    )

        return

    def get_embedding(self):
        if self.fst:
            embedding_fst = self.fst_u_embeddings.weight.cpu().data.numpy()
            embedding_fst /= np.sqrt(
                np.sum(embedding_fst * embedding_fst, 1)
            ).reshape(-1, 1)
        if self.snd:
            embedding_snd = self.snd_u_embeddings.weight.cpu().data.numpy()
            embedding_snd /= np.sqrt(
                np.sum(embedding_snd * embedding_snd, 1)
            ).reshape(-1, 1)
        if self.fst and self.snd:
            embedding = np.concatenate((embedding_fst, embedding_snd), 1)
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(
                -1, 1
            )
        elif self.fst and not self.snd:
            embedding = embedding_fst
        elif self.snd and not self.fst:
            embedding = embedding_snd
        else:
            pass

        return embedding

    def save_embedding(self, dataset, file_name):
        """Write embedding to local file. Only used when node ids are numbers.

        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.get_embedding()
        np.save(file_name, embedding)

    def save_embedding_pt(self, dataset, file_name):
        """For ogb leaderboard."""
        embedding = torch.Tensor(self.get_embedding()).cpu()
        embedding_empty = torch.zeros_like(embedding.data)
        valid_nodes = torch.LongTensor(dataset.valid_nodes)
        valid_embedding = embedding.data.index_select(0, valid_nodes)
        embedding_empty.index_add_(0, valid_nodes, valid_embedding)

        torch.save(embedding_empty, file_name)
