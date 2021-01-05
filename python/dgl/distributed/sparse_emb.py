"""Define sparse embedding and optimizer."""

from .. import backend as F
from .. import utils
from .dist_tensor import DistTensor
import abc
import torch as th

class DistEmbedding:
    '''Distributed embeddings.

    DGL provides a distributed embedding to support models that require learnable embeddings.
    DGL's distributed embeddings are mainly used for learning node embeddings of graph models.
    Because distributed embeddings are part of a model, they are updated by mini-batches.
    The distributed embeddings have to be updated by DGL's optimizers instead of
    the optimizers provided by the deep learning frameworks (e.g., Pytorch and MXNet).

    To support efficient training on a graph with many nodes, the embeddings support sparse
    updates. That is, only the embeddings involved in a mini-batch computation are updated.
    Currently, DGL provides only one optimizer: `SparseAdagrad`. DGL will provide more
    optimizers in the future.

    Distributed embeddings are sharded and stored in a cluster of machines in the same way as
    py:meth:`dgl.distributed.DistTensor`, except that distributed embeddings are trainable.
    Because distributed embeddings are sharded
    in the same way as nodes and edges of a distributed graph, it is usually much more
    efficient to access than the sparse embeddings provided by the deep learning frameworks.

    Parameters
    ----------
    num_embeddings : int
        The number of embeddings. Currently, the number of embeddings has to be the same as
        the number of nodes or the number of edges.
    embedding_dim : int
        The dimension size of embeddings.
    name : str, optional
        The name of the embeddings. The name can uniquely identify embeddings in a system
        so that another DistEmbedding object can referent to the embeddings.
    init_func : callable, optional
        The function to create the initial data. If the init function is not provided,
        the values of the embeddings are initialized to zero.
    part_policy : PartitionPolicy, optional
        The partition policy that assigns embeddings to different machines in the cluster.
        Currently, it only supports node partition policy or edge partition policy.
        The system determines the right partition policy automatically.

    Examples
    --------
    >>> def initializer(shape, dtype):
            arr = th.zeros(shape, dtype=dtype)
            arr.uniform_(-1, 1)
            return arr
    >>> emb = dgl.distributed.DistEmbedding(g.number_of_nodes(), 10, init_func=initializer)
    >>> optimizer = dgl.distributed.SparseAdagrad([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     feats = emb(nids)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()

    Note
    ----
    When a ``DistEmbedding``  object is used when the deep learning framework is recording
    the forward computation, users have to invoke py:meth:`~dgl.distributed.SparseAdagrad.step`
    afterwards. Otherwise, there will be some memory leak.
    '''
    def __init__(self, num_embeddings, embedding_dim, name=None,
                 init_func=None, part_policy=None):
        self._tensor = DistTensor((num_embeddings, embedding_dim), F.float32, name,
                                  init_func, part_policy)
        self._trace = []

    def __call__(self, idx):
        idx = utils.toindex(idx).tousertensor()
        emb = self._tensor[idx]
        if F.is_recording():
            emb = F.attach_grad(emb)
            self._trace.append((idx, emb))
        return emb

def _init_state(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class SparseAdagradUDF:
    ''' The UDF to update the embeddings with sparse Adagrad.

    Parameters
    ----------
    lr : float
        The learning rate.
    '''
    def __init__(self, lr):
        self._lr = lr

    def __call__(self, data_store, name, indices, grad):
        ''' Update the embeddings with sparse Adagrad.

        This function runs on the KVStore server. It updates the gradients by scaling them
        according to the state sum.

        Parameters
        ----------
        data_store : dict of data
            all data in the kvstore.
        name : str
            data name
        indices : tensor
            the indices in the local tensor.
        grad : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of id
        '''
        eps = 1e-6
        clr = self._lr

        embs = data_store[name]
        state = data_store[name + "_sum"]

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(indices, return_inverse=True, return_counts=True)
        grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=grad.device)
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)

        grad_sum = (grad_values * grad_values)
        state_dev = state.device
        state_idx = grad_indices.to(state_dev)
        grad_state = state[state_idx].to(grad.device)
        grad_state += grad_sum
        state[state_idx] = grad_state.to(state_dev)

        std_values = grad_state.add_(eps).sqrt_()
        tmp = clr * grad_values / std_values
        embs[state_idx] -= tmp.to(state_dev)

class SparseGradOptimizer(abc.ABC):
    r''' The abstract sparse optimizer.

    Parameters
    ----------
    params : list of GraphSparseEmbedding
        The list of GraphSparseEmbeddings.
    lr : float
        The learning rate.
    async_update : Bool
        Whether use asynchronize update
    '''
    def __init__(self, params, lr, async_update=False):
        self._params = params
        self._lr = lr
        self._async_update = async_update

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to update embeddings
        '''
        with th.no_grad():
            for emb in self._params:
                name = emb._tensor.name
                kvstore = emb._tensor.kvstore
                trace = emb._trace

                if len(trace) == 1:
                    kvstore.push(name, trace[0][0], F.grad(trace[0][1]))
                else:
                    idxs = [t[0] for t in trace]
                    grads = [F.grad(t[1]) for t in trace]
                    idxs = F.cat(idxs, 0)
                    # Here let's adjust the gradients with the learning rate first.
                    # We'll need to scale them with the state sum on the kvstore server
                    # after we push them.
                    grads = F.cat(grads, 0)
                    kvstore.push(name, idxs, grads)
                # Clean up the old traces.
                emb._trace = []

    def zero_grad(self):
        """ dummy
        """
        pass

class SparseAdagrad(SparseGradOptimizer):
    r''' The sparse Adagrad optimizer.

    This optimizer implements a sparse version of Adagrad algorithm for optimizing
    :func:`dgl.distributed.DistEmbedding`. In each mini-batch, it only updates the embeddings
    involved in the mini-batch to support efficient training on a graph with many
    nodes and edges.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    Parameters
    ----------
    params : list of DistEmbedding
        The list of DistEmbedding.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        super(SparseAdagrad, self).__init__(params, lr)
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, DistEmbedding), 'SparseAdagrad only supports DistEmbeding'
            name = emb._tensor.name
            kvstore = emb._tensor.kvstore
            policy = emb._tensor.part_policy
            kvstore.init_data(name + "_sum",
                              emb._tensor.shape, emb._tensor.dtype,
                              policy, _init_state)
            kvstore.register_push_handler(name, SparseAdagradUDF(self._lr))

class SparseAdamUDF:
    ''' The UDF to update the embeddings with sparse Adam.

    Parameters
    ----------
    lr : float
        The learning rate.
    '''
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2

    def __call__(self, data_store, name, indices, grad):
        """ Update embeddings in a sparse manner

        This function runs on the KVStore server. It updates the gradients by scaling them
        according to the state sum.

        Parameters
        ----------
        data_store : dict of data
            all data in the kvstore.
        name : str
            data name
        indices : tensor
            the indices in the local tensor.
        grad : tensor (torch.tensor)
            a tensor with the same row size of id
        """
        with F.no_grad():
            beta1 = self._beta1
            beta2= self._beta2
            eps = 1e-8

            embs = data_store[name]
            state_step = data_store[name + "_step"]
            state_mem = data_store[name + "_mean"]
            state_power = data_store[name + "_power"]

            clr = self._lr
            exec_dev = grad.device
            state_dev = state_step.device

            # the update is non-linear so indices must be unique
            grad_indices, inverse, cnt = th.unique(indices, return_inverse=True, return_counts=True)
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
            embs[state_idx] -= std_values.to(state_dev)

class SparseAdam(SparseGradOptimizer):
    r''' The sparse Adam optimizer.

    This optimizer implements a sparse version of Adam algorithm for optimizing
    :func:`dgl.distributed.DistEmbedding`. In each mini-batch, it only updates the embeddings
    involved in the mini-batch to support efficient training on a graph with many
    nodes and edges.

    Adam maintains a :math:`Gm_{t,i,j}` and `Gp_{t,i,j}` for every parameter in the embeddings, where
    :math:`Gm_{t,i,j}=beta1 * Gm_{t-1,i,j} + (1-beta1) * g_{t,i,j}`,
    :math:`Gp_{t,i,j}=beta2 * Gp_{t-1,i,j} + (1-beta2) * g_{t,i,j}^2`,
    :math:`g_{t,i,j} = lr * Gm_{t,i,j} / (1 - beta1^t) / \sqrt{Gp_{t,i,j} / (1 - beta2^t)}` and
    :math:`g_{t,i,j}` is the gradient of the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    Parameters
    ----------
    params : list of DistEmbedding
        The list of DistEmbedding.
    lr : float
        The learning rate.
    beta1 : float
        The beta1 of Adam.
    beta2 : float
        The beta2 of Adam.
    '''
    def __init__(self, params, lr, beta1=0.9, beta2=0.999):
        super(SparseAdam, self).__init__(params, lr)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, DistEmbedding), 'SparseAdam only supports DistEmbeding'
            name = emb._tensor.name
            kvstore = emb._tensor.kvstore
            policy = emb._tensor.part_policy
            kvstore.init_data(name + "_mean",
                              emb._tensor.shape, emb._tensor.dtype,
                              policy, _init_state)
            kvstore.init_data(name + "_power",
                              emb._tensor.shape, emb._tensor.dtype,
                              policy, _init_state)
            kvstore.init_data(name + "_step",
                              (emb._tensor.shape[0],), emb._tensor.dtype,
                              policy, _init_state)
            kvstore.register_push_handler(name, SparseAdamUDF(self._lr, self._beta1, self._beta2))
