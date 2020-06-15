"""Define sparse embedding and optimizer."""

from .. import backend as F

def _get_ndata_name(name):
    ''' This is to get the name of node data in the kvstore.

    KVStore doesn't understand node data or edge data. We'll use a prefix to distinguish them.
    '''
    return 'node:' + name

class SparseEmbedding:
    ''' Sparse embeddings in the distributed KVStore.

    The sparse embeddings are only used as node embeddings.

    Parameters
    ----------
    g : DistGraph
        The distributed graph object.
    name : str
        The name of the embeddings
    shape : tuple of int
        The shape of the embedding. The first dimension should be the number of nodes.
    initializer : callable
        The function to create the initial data.
    '''
    def __init__(self, g, name, shape, initializer):
        assert shape[0] == g.number_of_nodes()
        g._client.init_data(_get_ndata_name(name), shape, F.float32, 'node',
                            g.get_partition_book(), initializer)
        g._ndata._add(name)
        g._node_embs.append(self)

        self._tensor = g.ndata[name]
        self._trace = []

    def __call__(self, idx):
        emb = F.attach_grad(self._tensor[idx])
        self._trace.append((idx, emb))
        return emb

def sparse_adagrad_optimize(data_store, name, indices, data):
    ''' Update the embeddings with sparse Adagrad.

    This function runs on the KVStore server. It updates the gradients by scaling them
    according to the state sum. The gradients have been adjusted by the learning rate
    before being pushed to the kvstore.

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
    grad_sum = (grad_values * grad_values).mean(1)
    state_sum.index_add_(0, grad_indices, grad_sum)
    std = state_sum[grad_indices]  # _sparse_mask
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    embs.index_add_(0, grad_indices, grad_values / std_values)

def _init_state(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class SparseAdagrad:
    ''' The Adagrad optimizer for sparse embeddings.

    This optimizer collects gradients for the sparse embeddings and update
    the embeddings in the distributed KVStore.

    Parameters
    ----------
    params : list of SparseEmbeddings
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
                              (emb._tensor.shape[0],), emb._tensor.dtype,
                              policy.policy_str, policy.partition_book, _init_state)
            kvstore.register_push_handler(name, sparse_adagrad_optimize)

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the sparse embeddings to the distributed kvstore and update the embeddings
        in the kvstore.
        '''
        for emb in self._params:
            name = emb._tensor.name
            kvstore = emb._tensor.kvstore
            trace = emb._trace
            if len(trace) == 1:
                kvstore.push(name, trace[0][0], trace[0][1].grad.data)
            else:
                # TODO(zhengda) we need to merge the gradients of the same embeddings first.
                idxs = [t[0] for t in trace]
                grads = [t[1].grad.data for t in trace]
                idxs = F.cat(idxs, 0)
                # Here let's adjust the gradients with the learning rate first.
                # We'll need to scale them with the state sum on the kvstore server
                # after we push them.
                grads = F.cat(grads, 0) * -self._lr
                kvstore.push(name, idxs, grads)
            # Clean up the old traces.
            emb._trace = []
