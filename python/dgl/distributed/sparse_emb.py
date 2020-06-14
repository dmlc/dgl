"""Define sparse embedding and optimizer."""

from .. import backend as F

class SparseEmbedding:
    def __init__(self, g, name, embedding_dim):
        # TODO I need to initialize embeddings.
        g.init_node_emb(name, (g.number_of_nodes(), embedding_dim))
        self._tensor = g.ndata[name]
        self._trace = []

    def __call__(self, idx):
        # This is pytorch way.
        emb = self._tensor[idx].requires_grad_(True)
        self.trace.append((idx, emb))
        return emb

    def reset_parameters(self):
        pass

def sparse_adagrad_optimize(name, ID, data, target):
    ''' Update the embeddings with sparse Adagrad.

    This function runs on the KVStore server. It updates the gradients by scaling them
    according to the state sum. The gradients have been adjusted by the learning rate
    before being pushed to the kvstore.

    Parameters
    ----------
    name : str
        data name
    ID : tensor
        a vector storing the ID list.
    data : tensor (mx.ndarray or torch.tensor)
        a tensor with the same row size of id
    target : dict of data
        all data in the kvstore.
    '''
    # TODO are all indices local?
    grad_indices = ID
    grad_values = data
    embs = target[name]
    state_sum = target[name + "_sum"]
    grad_sum = (grad_values * grad_values).mean(1)
    state_sum.index_add_(0, grad_indices, grad_sum)
    std = state_sum[grad_indices]  # _sparse_mask
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    embs.index_add_(0, grad_indices, grad_values / std_values)

def init_state(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class SparseAdagrad:
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            name = emb._tensor.name
            kv = emb._tensor.kvstore
            policy = emb._tensor.part_policy
            kv.init_data(name + "_sum", (emb._tensor.shape[0],), emb._tensor.dtype,
                         policy.policy_str, policy.partition_book, init_state)
        kv.register_push_handler(sparse_adagrad_optimize)

    def step(self):
        for emb in self._params:
            name = emb._tensor.name
            kv = emb._tensor.kvstore
            trace = emb.trace
            if len(trace) == 1:
                kv.push(name, trace[0][0], trace[0][1].grad.data)
            else:
                # TODO(zhengda) we need to merge the gradients of the same embeddings first.
                idxs = [t[0] for t in trace]
                grads = [t[1].grad.data for t in trace]
                idxs = F.cat(idxs, 0)
                # Here let's adjust the gradients with the learning rate first.
                # We'll need to scale them with the state sum on the kvstore server after we push them.
                grads = F.cat(grads, 0) * -self._lr
                kv.push(name, idxs, grads)
            # Clean up the old traces.
            emb.trace = []
