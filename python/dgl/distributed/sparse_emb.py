"""Define sparse embedding and optimizer."""

from .. import backend as F
from .. import utils
from .dist_tensor import DistTensor
import torch.distributed as dist
import torch as th

class NodeEmbedding:
    '''Distributed node embeddings.

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
        so that another NodeEmbedding object can referent to the same embeddings.
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
    >>> emb = dgl.distributed.NodeEmbedding(g.number_of_nodes(), 10, init_func=initializer)
    >>> optimizer = dgl.distributed.optim.SparseAdagrad([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     feats = emb(nids)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()

    Note
    ----
    When a ``NodeEmbedding``  object is used when the deep learning framework is recording
    the forward computation, users have to invoke py:meth:`~dgl.distributed.optim.SparseAdagrad.step`
    afterwards. Otherwise, there will be some memory leak.
    '''
    def __init__(self, num_embeddings, embedding_dim, name=None,
                 init_func=None, part_policy=None):
        self._tensor = DistTensor((num_embeddings, embedding_dim), F.float32, name,
                                  init_func=init_func)
        self._trace = []
        self._name = name
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        # Check whether it is multi-gpu training or not.
        if th.distributed.is_initialized():
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
        else:
            assert 'th.distributed shoud be initialized'
        self._optm_state = None # track optimizer state
        self._part_policy = part_policy

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

    @property
    def part_policy(self):
        return self._part_policy

    @property
    def rank(self):
        """Return rank of current process.

        Returns
        -------
        int
            The rank of current process.
        """
        return self._rank

    @property
    def world_size(self):
        """Return world size of the pytorch distributed training env.

        Returns
        -------
        int
            The world size of the pytorch distributed training env.
        """
        return self._world_size

    @property
    def name(self):
        return self._tensor._name

    @property
    def kvstore(self):
        return self._tensor.kvstore

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def set_optm_state(self, state):
        """Store the optimizer related state tensor.

        Parameters
        ----------
        state : tuple of torch.Tensor
            Optimizer related state.
        """
        self._optm_state = state

    @property
    def optm_state(self):
        """Return the optimizer related state tensor.

        Returns
        -------
        tuple of torch.Tensor
            The optimizer related state.
        """
        return self._optm_state

