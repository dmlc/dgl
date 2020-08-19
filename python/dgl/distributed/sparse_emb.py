"""Define sparse embedding and optimizer."""

from .. import backend as F
from .. import utils
from .dist_tensor import DistTensor

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

class SparseAdagradUDF:
    ''' The UDF to update the embeddings with sparse Adagrad.

    Parameters
    ----------
    lr : float
        The learning rate.
    '''
    def __init__(self, lr):
        self._lr = lr

    def __call__(self, data_store, name, indices, data):
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
        data : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of id
        '''
        grad_indices = indices
        grad_values = data
        embs = data_store[name]
        state_sum = data_store[name + "_sum"]
        with F.no_grad():
            grad_sum = F.mean(grad_values * grad_values, 1)
            F.index_add_inplace(state_sum, grad_indices, grad_sum)
            std = state_sum[grad_indices]  # _sparse_mask
            std_values = F.unsqueeze((F.sqrt(std) + 1e-10), 1)
            F.index_add_inplace(embs, grad_indices, grad_values / std_values * (-self._lr))

def _init_state(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class SparseAdagrad:
    r''' The sparse Adagrad optimizer.

    This optimizer implements a lightweight version of Adagrad algorithm for optimizing
    :func:`dgl.distributed.DistEmbedding`. In each mini-batch, it only updates the embeddings
    involved in the mini-batch to support efficient training on a graph with many
    nodes and edges.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    Instead of maintaining :math:`G_{t,i,j}`, this implementation maintains :math:`G_{t,i}`
    for every embedding :math:`i`:

    .. math::
      G_{t,i}=G_{t-1,i}+ \frac{1}{p} \sum_{0 \le j \lt p}g_{t,i,j}^2

    where :math:`p` is the dimension size of an embedding.

    The benefit of the implementation is that it consumes much smaller memory and runs
    much faster if users' model requires learnable embeddings for nodes or edges.

    Parameters
    ----------
    params : list of DistEmbeddings
        The list of distributed embeddings.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, DistEmbedding), 'SparseAdagrad only supports DistEmbeding'
            name = emb._tensor.name
            kvstore = emb._tensor.kvstore
            policy = emb._tensor.part_policy
            kvstore.init_data(name + "_sum",
                              (emb._tensor.shape[0],), emb._tensor.dtype,
                              policy, _init_state)
            kvstore.register_push_handler(name, SparseAdagradUDF(self._lr))

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the embeddings involved in a mini-batch to DGL's servers and update the embeddings.
        '''
        with F.no_grad():
            for emb in self._params:
                name = emb._tensor.name
                kvstore = emb._tensor.kvstore
                trace = emb._trace
                if len(trace) == 1:
                    kvstore.push(name, trace[0][0], F.grad(trace[0][1]))
                else:
                    # TODO(zhengda) we need to merge the gradients of the same embeddings first.
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
