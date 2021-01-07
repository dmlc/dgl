import torch as th
import abc
from abc import abstractmethod
from .tensor import attach_grad, is_recording, data_type_dict, zerocopy_to_dlpack, zerocopy_from_dlpack
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from datetime import timedelta

from ... import ndarray as nd
from ..._ffi.ndarray import empty_shared_mem

dtype_dict = data_type_dict()
dtype_dict = {dtype_dict[key]:key for key in dtype_dict}

_store = None

def _get_shared_mem_array(name, shape, dtype):
    new_arr = empty_shared_mem(name, False, shape, dtype_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return zerocopy_from_dlpack(dlpack)

def _create_shared_mem_array(name, shape, dtype):
    new_arr = empty_shared_mem(name, True, shape, dtype_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return zerocopy_from_dlpack(dlpack)

class GraphSparseQueues:
    ''' Queues for sparse embeddings metadata synchronization

    Parameters
    ----------
    world_size: int
        Number of gpu processes
    '''
    def __init__(self, world_size):
        self._queues = [mp.Queue() for _ in range(world_size)]

    @property
    def queues(self):
        return self._queues

class GraphSparseEmbedding:
    '''Sparse embeddings for graph.

    DGL provides a sparse embedding to support models that require learnable embeddings.
    DGL's sparse embeddings are mainly used for learning node embeddings of graph models.
    The sparse embeddings have to be updated by DGL's optimizers instead of
    the optimizers provided by the deep learning frameworks (e.g., Pytorch).

    To support efficient training on a graph with many nodes, the embeddings support sparse
    updates. That is, only the embeddings involved in a mini-batch computation are updated.
    Currently, DGL provides two optimizer: `SparseAdagrad` and `SparseAdam. DGL will provide more
    optimizers in the future.

    Parameters
    ----------
    num_embeddings : int
        The number of embeddings. Currently, the number of embeddings has to be the same as
        the number of nodes or the number of edges.
    embedding_dim : int
        The dimension size of embeddings.
    device : th.device
        Target device to manipulate embedding related opations
    name : str
        The name of the embeddings. The name should uniquely identify embeddings in a system.
    init_func : callable, optional
        The function to create the initial data. If the init function is not provided,
        the values of the embeddings are initialized to zero.
    group : str, optional
        Group name.

    Examples
    --------
    Before launching multiple gpu processes

    >>> def initializer(shape, dtype):
            arr = th.zeros(shape, dtype=dtype)
            arr.uniform_(-1, 1)
            return arr

    In each multiple gpu process

    >>> emb = dgl.GraphSparseEmbedding(g.number_of_nodes(), 10, init_func=initializer,
        rank=rank, world_size=num_gpus)
    >>> optimizer = dgl.SparseAdagradOptimizer([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     feats = emb(nids)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()
    '''

    def __init__(self, num_embeddings, embedding_dim, name, device=th.device('cpu'),
                 init_func=None, group=None, host_name='127.0.0.1', port=12346):
        global _store
        if th.distributed.is_initialized():
            rank = th.distributed.get_rank() if group is None else th.distributed.get_rank(group)
            world_size = th.distributed.get_world_size() if group is None else th.distributed.get_world_size(group)
        else:
            rank = -1
            world_size = 0
        self._rank = rank
        self._world_size = world_size
        self._device = device

        if rank <= 0:
            emb = _create_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)
            if init_func is not None:
                emb = init_func(emb)
        if rank == 0:
            if world_size > 1:
                if _store is None:
                    _store = th.distributed.TCPStore(host_name, port, world_size, True, timedelta(seconds=30))
            for i in range(1, world_size):
                # send embs
                _store.set(name, name)
        elif rank > 0:
            # receive
            if _store is None:
                _store = th.distributed.TCPStore(host_name, port, world_size, False, timedelta(seconds=30))
            _store.wait([name])
            emb = _get_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)

        self._tensor = emb
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._name = name
        self._state = None
        self._trace = []

    def __call__(self, idx):
        emb = self._tensor[idx].to(self._device)
        if is_recording():
            emb = attach_grad(emb)
            self._trace.append((idx.to(self._device, non_blocking=True), emb))
        return emb

    @property
    def rank(self):
        return self._rank

    @property
    def name(self):
        return self._name


    @property
    def world_size(self):
        return self._world_size

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def kvstore(self):
        return self._store

    def set_opt_state(self, state):
        self._state = state

    @property
    def opt_state(self):
        return self._state

    @property
    def trace(self):
        return self._trace

    def reset_trace(self):
        self._trace = []

    @property
    def emb_tensor(self):
        return self._tensor

class SparseGradOptimizer(abc.ABC):
    r''' The abstract sparse optimizer.

    Parameters
    ----------
    params : list of GraphSparseEmbedding
        The list of GraphSparseEmbeddings.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._shared_cache = {}

    def step(self):
        global _store
        ''' The step function.

        The step function is invoked at the end of every batch to update embeddings
        '''
        with th.no_grad():
            update_embs = {emb.name: ([],[]) for emb in self._params}
            for emb in self._params:
                num_embeddings = emb.num_embeddings
                emb_name = emb.name
                range_size = (num_embeddings + self._world_size - 1) // self._world_size \
                    if self._world_size > 0 else 0
                for idx, data in emb._trace:
                    grad = data.grad.data
                    device = grad.device
                    idx_dtype = idx.dtype
                    grad_dtype = grad.dtype
                    grad_dim = grad.shape[1]

                    if self._world_size > 0:
                        if emb_name not in self._shared_cache:
                            self._shared_cache[emb_name] = {}
                        idx_list = []
                        grad_list = []
                        for i in range(self._world_size):
                            start = i * range_size
                            end = (i + 1) * range_size \
                                if (i + 1) * range_size < num_embeddings \
                                else num_embeddings
                            if i == 0:
                                mask = idx < end
                            elif i + 1 == self._world_size:
                                mask = idx >= start
                            else:
                                mask = th.logical_and((idx >= start),(idx < end))
                            idx_i = idx[mask]
                            grad_i = grad[mask]

                            if i == self._rank:
                                update_embs[emb_name][0].append(idx_i)
                                update_embs[emb_name][1].append(grad_i)
                            else:
                                idx_i = idx_i.to(th.device('cpu'))
                                grad_i = grad_i.to(th.device('cpu'))
                                idx_shmem_name = 'idx_{}_{}_{}'.format(emb_name, self._rank, i)
                                grad_shmem_name = 'grad_{}_{}_{}'.format(emb_name, self._rank, i)
                                if idx_shmem_name not in self._shared_cache[emb_name] or \
                                    self._shared_cache[emb_name][idx_shmem_name].shape[0] < idx_i.shape[0]:
                                    idx_shmem = _create_shared_mem_array(idx_shmem_name,
                                        (idx_i.shape[0] * 2 + 2,), idx_dtype)
                                    grad_shmem = _create_shared_mem_array(grad_shmem_name,
                                        (idx_i.shape[0] * 2 + 2, grad_dim), grad_dtype)
                                    self._shared_cache[emb_name][idx_shmem_name] = idx_shmem
                                    self._shared_cache[emb_name][grad_shmem_name] = grad_shmem

                                self._shared_cache[emb_name][idx_shmem_name][:idx_i.shape[0]] = idx_i
                                self._shared_cache[emb_name][grad_shmem_name][:idx_i.shape[0]] = grad_i
                                _store.set(idx_shmem_name, str(idx_i.shape[0]))

                        wait_keys = []
                        for i in range(self._world_size):
                            if i != self._rank:
                                idx_shmem_name = 'idx_{}_{}_{}'.format(emb_name, i, self._rank)
                                grad_shmem_name = 'grad_{}_{}_{}'.format(emb_name, i, self._rank)
                                size = int(_store.get(idx_shmem_name))
                                if idx_shmem_name not in self._shared_cache[emb_name] or \
                                    self._shared_cache[emb_name][idx_shmem_name].shape[0] < size:
                                    idx_shmem = _get_shared_mem_array(idx_shmem_name,
                                        (size * 2 + 2,), idx_dtype)
                                    grad_shmem = _get_shared_mem_array(grad_shmem_name,
                                        (size * 2 + 2, grad_dim), grad_dtype)
                                    self._shared_cache[emb_name][idx_shmem_name] = idx_shmem
                                    self._shared_cache[emb_name][grad_shmem_name] = grad_shmem
                                idx_i = self._shared_cache[emb_name][idx_shmem_name][:size]
                                grad_i = self._shared_cache[emb_name][grad_shmem_name][:size]
                                update_embs[emb_name][0].append(idx_i.to(device, non_blocking=True))
                                update_embs[emb_name][1].append(grad_i.to(device, non_blocking=True))
                    else:
                        update_embs[emb_name][0].append(idx)
                        update_embs[emb_name][1].append(grad)
                emb.reset_trace()

            for emb in self._params:
                emb_name = emb.name

                idx = th.cat(update_embs[emb_name][0], dim=0)
                grad = th.cat(update_embs[emb_name][1], dim=0)
                self.update(idx, grad, emb)

            if self._world_size > 1:
                th.distributed.barrier()

    @abstractmethod
    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : GraphSparseEmbedding
            Sparse embedding to update.
        """
        pass

    def zero_grad(self):
        """ dummy
        """
        pass

class SparseAdagradOptimizer(SparseGradOptimizer):
    r''' The sparse Adagrad optimizer.

    This optimizer implements a sparse version of Adagrad algorithm for optimizing
    :func:`dgl.backend.GraphSparseEmbedding`. In each mini-batch, it only updates the embeddings
    involved in the mini-batch to support efficient training on a graph with many
    nodes and edges.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    Parameters
    ----------
    params : list of GraphSparseEmbedding
        The list of GraphSparseEmbeddings.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        super(SparseAdagradOptimizer, self).__init__(params, lr)
        self._rank = None
        self._world_size = None
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, GraphSparseEmbedding), 'SparseAdagradOptimizer only supports GraphSparseEmbedding'

            if self._rank is None:
                self._rank = emb.rank
                self._world_size = emb.world_size
            else:
                assert self._rank == emb.rank, 'MultiGPU rank for each embedding should be same.'
                assert self._world_size == emb.world_size, 'MultiGPU world_size for each embedding should be same.'
            if self._rank <= 0:
                state = emb.new().resize_(emb.shape).zero_()
            if self._rank == 0:
                state.share_memory_()
                for i in range(1, world_size):
                    # send embs
                    emb.queues[i].put(state)
            elif self._rank > 0:
                # receive
                state = emb.queues[self._rank].get()
            emb.set_opt_state(state)

    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : GraphSparseEmbedding
            Sparse embedding to update.
        """
        eps = 1e-6
        clr = self._lr

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(idx, return_inverse=True, return_counts=True)
        grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=grad.device)
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)

        grad_sum = (grad_values * grad_values)
        state = emb.opt_state
        state_dev = state.device
        state_idx = grad_indices.to(state_dev)
        grad_state = state[state_idx].to(grad.device)
        grad_state += grad_sum
        state[state_idx] = grad_state.to(state_dev)

        std_values = grad_state.add_(eps).sqrt_()
        tmp = clr * grad_values / std_values
        emb.emb_tensor[state_idx] -= tmp.to(state_dev)

class SparseAdamOptimizer(SparseGradOptimizer):
    r''' The sparse Adam optimizer.

    This optimizer implements a sparse version of Adam algorithm for optimizing
    :func:`dgl.backend.GraphSparseEmbedding`. In each mini-batch, it only updates the embeddings
    involved in the mini-batch to support efficient training on a graph with many
    nodes and edges.

    Adam maintains a :math:`Gm_{t,i,j}` and `Gp_{t,i,j}` for every parameter in the embeddings, where
    :math:`Gm_{t,i,j}=beta1 * Gm_{t-1,i,j} + (1-beta1) * g_{t,i,j}`,
    :math:`Gp_{t,i,j}=beta2 * Gp_{t-1,i,j} + (1-beta2) * g_{t,i,j}^2`,
    :math:`g_{t,i,j} = lr * Gm_{t,i,j} / (1 - beta1^t) / \sqrt{Gp_{t,i,j} / (1 - beta2^t)}` and
    :math:`g_{t,i,j}` is the gradient of the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    Parameters
    ----------
    params : list of GraphSparseEmbedding
        The list of GraphSparseEmbeddings.
    lr : float
        The learning rate.
    beta1 : float
        The beta1 of Adam.
    beta2 : float
        The beta2 of Adam.
    '''
    def __init__(self, params, lr, beta1=0.9, beta2=0.999):
        super(SparseAdamOptimizer, self).__init__(params, lr)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._rank = None
        self._world_size = None
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, GraphSparseEmbedding), 'SparseAdamOptimizer only supports GraphSparseEmbedding'

            if self._rank is None:
                self._rank = emb.rank
                self._world_size = emb.world_size
            else:
                assert self._rank == emb.rank, 'MultiGPU rank for each embedding should be same.'
                assert self._world_size == emb.world_size, 'MultiGPU world_size for each embedding should be same.'
            if self._rank <= 0:
                # may overflow if steps > 2B
                emb_name = emb.name
                state_step = _create_shared_mem_array(emb_name+'_step', (emb.emb_tensor.shape[0],), th.int).zero_()
                state_mem = _create_shared_mem_array(emb_name+'_mem', emb.emb_tensor.shape, th.float32).zero_()
                state_power = _create_shared_mem_array(emb_name+'_power', emb.emb_tensor.shape, th.float32).zero_()
            if self._rank == 0:
                state = (state_step, state_mem, state_power)
                emb_name = emb.name
                for i in range(1, self._world_size):
                    # send embs
                    _store.set(emb_name+'_opt', emb_name)
            elif self._rank > 0:
                # receive
                emb_name = emb.name
                _store.wait([emb_name+'_opt'])
                state_step = _get_shared_mem_array(emb_name+'_step', (emb.emb_tensor.shape[0],), th.int)
                state_mem = _get_shared_mem_array(emb_name+'_mem', emb.emb_tensor.shape, th.float32)
                state_power = _get_shared_mem_array(emb_name+'_power', emb.emb_tensor.shape, th.float32)

            state = (state_step, state_mem, state_power)
            emb.set_opt_state(state)

    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : GraphSparseEmbedding
            Sparse embedding to update.
        """
        with th.no_grad():
            beta1 = self._beta1
            beta2= self._beta2
            eps = 1e-8

            clr = self._lr
            state_step, state_mem, state_power = emb.opt_state
            exec_dev = grad.device
            state_dev = state_step.device

            # the update is non-linear so indices must be unique
            grad_indices, inverse, cnt = th.unique(idx, return_inverse=True, return_counts=True)
            state_idx = grad_indices.to(state_dev)
            state_step[state_idx] += 1
            state_step = state_step[state_idx].to(exec_dev, non_blocking=True)
            orig_mem = state_mem[state_idx].to(exec_dev, non_blocking=True)
            orig_power = state_power[state_idx].to(exec_dev, non_blocking=True)

            grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=exec_dev)
            grad_values.index_add_(0, inverse, grad)
            grad_values = grad_values / cnt.unsqueeze(1)

            grad_mem = grad_values
            grad_power = grad_values * grad_values
            update_mem = beta1 * orig_mem + (1-beta1) * grad_mem
            update_power = beta2 * orig_power + (1-beta2) * grad_power

            beta1 = th.pow(beta1, state_step)
            beta2 = th.pow(beta2, state_step)
            update_mem_corr = update_mem / (1 - beta1).unsqueeze(1)
            update_power_corr = update_power / (1 - beta2).unsqueeze(1)
            std_values = clr * update_mem_corr / (th.sqrt(update_power_corr) + eps)

            state_mem[state_idx] = update_mem.to(state_dev)
            state_power[state_idx] = update_power.to(state_dev)
            emb.emb_tensor[state_idx] -= std_values.to(state_dev)
