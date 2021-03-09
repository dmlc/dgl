"""Define sparse embedding and optimizer."""

from .. import backend as F
from .. import utils
from .dist_tensor import DistTensor
import torch.distributed as dist
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

    def reset_trace(self):
        '''Reset the traced data.
        '''
        self._trace = []

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
    def __init__(self, params, lr, eps=1e-10):
        self._params = params
        self._lr = lr
        self._eps = eps
        self._clean_grad = False
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, DistEmbedding), 'SparseAdagrad only supports DistEmbeding'
            name = emb._tensor.name
            kvstore = emb._tensor.kvstore
            policy = emb._tensor.part_policy
            kvstore.init_data(name + "_sum",
                              emb._tensor.shape, emb._tensor.dtype,
                              policy, _init_state)

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the embeddings involved in a mini-batch to DGL's servers and update the embeddings.
        '''
        with F.no_grad():
            shared_emb = {emb.name: ([], []) for emb in self._params}
            for emb in self._params:
                name = emb._tensor.name
                kvstore = emb._tensor.kvstore
                trace = emb._trace

                idics = [t[0] for t in trace]
                grads = [F.grad(t[1]) for t in trace]
                idics = F.cat(idics, 0)
                grads = F.cat(grads, 0)

                # will send grad to each corresponding trainer
                if kvstore.num_servers > 1:
                    # get idx split from kvstore
                    idx_split = kvstore.get_partid(name, idics)
                    idx_split_size = []
                    idics_list = []
                    grad_list = []
                    # split idx and grad first
                    for i in range(kvstore.num_servers):
                        mask = idx_split == i
                        idx_i = idics[mask]
                        grad_i = grads[mask]
                        idx_split_size.append(idx_i.shape[0])
                        idics_list.append(idx_i)
                        grad_list.append(grad_i)

                    # use scatter to sync across trainers about the p2p tensor size
                    # Note: If we have GPU nccl support, we can use all_to_all to sync information here
                    gather_list = [th.empty((1,), dtype=th.long) for _ in range(kvstore.num_servers)]
                    for i in range(kvstore.num_servers):
                        dist.scatter(gather_list[i], idx_split_size if i == kvstore.machine_id else [], src=kvstore.machine_id)

                    # Note: We may use nccl alltoallv to simplify this
                    # send tensor to each target trainer using torch.distributed.isend
                    # isend is async
                    for i in range(kvstore.num_servers):
                        if i == kvstore.machine_id:
                            shared_emb[name][0].append(idics_list[i])
                            shared_emb[name][1].append(grad_list[i])
                        else:
                            dist.isend(idics_list[i], dst=i)
                            dist.isend(grad_list[i], dst=i)

                    # receive tensor from all remote trainer using torch.distributed.recv
                    # recv is sync
                    for i in range(kvstore.num_servsers):
                        if i != kvstore.machine_id:
                            idx_recv = th.empty((gather_list[i],), dtype=idics.dtype)
                            grad_recv = th.empty((gather_list[i], grads.shape[1]), dtype=grads.dtype)
                            dist.recv(idx_recv, src=i)
                            dist.recv(grad_recv, src=i)

                            shared_emb[name][0].append(idx_recv)
                            shared_emb[name][1].append(grad_recv)

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            # do local update
            for emb in self._params:
                name = emb._tensor.name

                idx = th.cat(shared_emb[name][0], dim=0)
                grad = th.cat(shared_emb[name][1], dim=0)
                self.update(idx, grad, emb)


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
        emb : dgl.nn.NodeEmbedding
            Sparse embedding to update.
        """
        eps = self._eps
        clr = self._lr
        name = emb._tensor.name
        kvstore = emb._tensor.kvstore

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(idx, return_inverse=True, return_counts=True)
        grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=grad.device)
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        grad_sum = (grad_values * grad_values)

        # update grad state
        grad_state = kvstore.pull(name=name + "_sum", grad_indices)
        grad_state += grad_sum
        kvstore.push(name=name + "_sum", grad_indices, grad_state)

        # update emb
        std_values = grad_state.add_(eps).sqrt_()
        tmp = clr * grad_values / std_values
        emb = kvstore.pull(name=name, grad_indices)
        emb -= tmp
        kvstore.push(name=name + "_sum", grad_indices, emb)

    def zero_grad(self):
        """clean grad cache
        """
        self._clean_grad = True
