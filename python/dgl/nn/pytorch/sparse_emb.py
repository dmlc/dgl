"""Torch NodeEmbedding."""
from datetime import timedelta
import torch as th
from ...backend import pytorch as F
from ...utils import get_shared_mem_array, create_shared_mem_array

_STORE = None

class NodeEmbedding: # NodeEmbedding
    '''Class for storing node embeddings.

    The class is optimized for training large-scale node embeddings. It updates the embedding in
    a sparse way and can scale to graphs with millions of nodes. It also supports partitioning
    to multiple GPUs (on a single machine) for more acceleration. It does not support partitioning
    across machines.

    Currently, DGL provides two optimizers that work with this NodeEmbedding
    class: ``SparseAdagrad`` and ``SparseAdam``.

    The implementation is based on torch.distributed package. It depends on the pytorch
    default distributed process group to collect multi-process information and uses
    ``torch.distributed.TCPStore`` to share meta-data information across multiple gpu processes.
    It use the local address of '127.0.0.1:12346' to initialize the TCPStore.

    NOTE: The support of NodeEmbedding is experimental.

    Parameters
    ----------
    num_embeddings : int
        The number of embeddings. Currently, the number of embeddings has to be the same as
        the number of nodes.
    embedding_dim : int
        The dimension size of embeddings.
    name : str
        The name of the embeddings. The name should uniquely identify the embeddings in the system.
    init_func : callable, optional
        The function to create the initial data. If the init function is not provided,
        the values of the embeddings are initialized to zero.

    Examples
    --------
    Before launching multiple gpu processes

    >>> def initializer(emb):
            th.nn.init.xavier_uniform_(emb)
            return emb

    In each training process

    >>> emb = dgl.nn.NodeEmbedding(g.number_of_nodes(), 10, 'emb', init_func=initializer)
    >>> optimizer = dgl.optim.SparseAdam([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     ...
    ...     feats = emb(nids, gpu_0)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()
    '''

    def __init__(self, num_embeddings, embedding_dim, name,
                 init_func=None):
        global _STORE

        # Check whether it is multi-gpu training or not.
        if th.distributed.is_initialized():
            rank = th.distributed.get_rank()
            world_size = th.distributed.get_world_size()
        else:
            rank = -1
            world_size = 0
        self._rank = rank
        self._world_size = world_size
        host_name = '127.0.0.1'
        port = 12346

        if rank <= 0:
            emb = create_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)
            if init_func is not None:
                emb = init_func(emb)
        if rank == 0: # the master gpu process
            # for multi-gpu training, setup a TCPStore for
            # embeding status synchronization across GPU processes
            if _STORE is None:
                _STORE = th.distributed.TCPStore(
                    host_name, port, world_size, True, timedelta(seconds=30))
            for _ in range(1, world_size):
                # send embs
                _STORE.set(name, name)
        elif rank > 0:
            # receive
            if _STORE is None:
                _STORE = th.distributed.TCPStore(
                    host_name, port, world_size, False, timedelta(seconds=30))
            _STORE.wait([name])
            emb = get_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)

        self._store = _STORE
        self._tensor = emb
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._name = name
        self._optm_state = None # track optimizer state
        self._trace = [] # track minibatch

    def __call__(self, node_ids, device=th.device('cpu')):
        """
        node_ids : th.tensor
            Index of the embeddings to collect.
        device : th.device
            Target device to put the collected embeddings.
        """
        emb = self._tensor[node_ids].to(device)
        if F.is_recording():
            emb = F.attach_grad(emb)
            self._trace.append((node_ids.to(device, non_blocking=True), emb))
        return emb

    @property
    def store(self):
        """Return torch.distributed.TCPStore for
        meta data sharing across processes.

        Returns
        -------
        torch.distributed.TCPStore
            KVStore used for meta data sharing.
        """
        return self._store

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
        """Return the name of NodeEmbedding.

        Returns
        -------
        str
            The name of NodeEmbedding.
        """
        return self._name

    @property
    def num_embeddings(self):
        """Return the number of embeddings.

        Returns
        -------
        int
            The number of embeddings.
        """
        return self._num_embeddings

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

    @property
    def trace(self):
        """Return a trace of the indices of embeddings
        used in the training step(s).

        Returns
        -------
        [torch.Tensor]
            The indices of embeddings used in the training step(s).
        """
        return self._trace

    def reset_trace(self):
        """Clean up the trace of the indices of embeddings
        used in the training step(s).
        """
        self._trace = []

    @property
    def emb_tensor(self):
        """Return the tensor storing the node embeddings

        Returns
        -------
        torch.Tensor
            The tensor storing the node embeddings
        """
        return self._tensor
