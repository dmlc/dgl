"""Define sparse embedding and optimizer."""

from .. import backend as F
from .. import utils
from .dist_tensor import DistTensor

class DistEmbedding:
    '''Embeddings in the distributed training.

    Parameters
    ----------
    num_embeddings : int
        The number of embeddings
    embedding_dim : int
        The dimension size of embeddings.
    name : str
        The name of the embeddings
    init_func : callable
        The function to create the initial data.
    part_policy : PartitionPolicy
        The partition policy.

    Examples
    --------
    >>> emb_init = lambda shape, dtype: F.zeros(shape, dtype, F.cpu())
    >>> emb = dgl.distributed.DistEmbedding(g.number_of_nodes(), 10)
    >>> optimizer = dgl.distributed.SparseAdagrad([emb], lr=0.001)
    >>> for blocks in dataloader:
    >>>     feats = emb(nids)
    >>>     loss = F.sum(feats + 1, 0)
    >>>     loss.backward()
    >>>     optimizer.step()
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
            grad_sum = grad_values * grad_values
            # This is faster than F.index_add_inplace. As long as the indices are unique,
            # += has the same behavior as F.index_add_inplace. += is much faster.
            state_sum[grad_indices] += grad_sum
            std = state_sum[grad_indices]  # _sparse_mask
            std_values = F.sqrt(std) + 1e-10
            F.index_add_inplace(embs, grad_indices, grad_values / std_values * (-self._lr))

def _init_state(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class SparseAdagrad:
    ''' The Adagrad optimizer for sparse embeddings.

    This optimizer collects gradients for the sparse embeddings and update
    the embeddings in the distributed KVStore.

    Parameters
    ----------
    params : list of DistEmbeddings
        The list of sparse embeddings.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            name = emb._tensor.name
            kvstore = emb._tensor.kvstore
            policy = emb._tensor.part_policy
            kvstore.init_data(name + "_sum",
                              emb._tensor.shape, emb._tensor.dtype,
                              policy, _init_state)
            kvstore.register_push_handler(name, SparseAdagradUDF(self._lr))

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the sparse embeddings to the distributed kvstore and update the embeddings
        in the kvstore.
        '''
        with F.no_grad():
            for emb in self._params:
                name = emb._tensor.name
                kvstore = emb._tensor.kvstore
                trace = emb._trace
                if len(trace) == 1:
                    idxs = trace[0][0]
                    grads = F.grad(trace[0][1])
                else:
                    # TODO(zhengda) we need to merge the gradients of the same embeddings first.
                    idxs = [t[0] for t in trace]
                    grads = [F.grad(t[1]) for t in trace]
                    idxs = F.cat(idxs, 0)
                    # Here let's adjust the gradients with the learning rate first.
                    # We'll need to scale them with the state sum on the kvstore server
                    # after we push them.
                    grads = F.cat(grads, 0)

                uniq_idxs, inverse_idxs = F.unique(idxs, return_inverse=True)
                if len(uniq_idxs) != len(idxs):
                    shape = list(grads.shape)
                    shape[0] = len(uniq_idxs)
                    coalesced_grads = F.zeros(shape, F.dtype(grads), F.cpu())
                    F.index_add_inplace(coalesced_grads, inverse_idxs, grads)
                    idxs = uniq_idxs
                    grads = coalesced_grads
                kvstore.push(name, idxs, grads)
                # Clean up the old traces.
                emb._trace = []
