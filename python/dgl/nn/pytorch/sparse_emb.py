import torch as th
from ...backend import pytorch as F
from ...utils import get_shared_mem_array, create_shared_mem_array
from datetime import timedelta

from ... import ndarray as nd

_store = None

class NodeEmbedding: # NodeEmbedding
    '''Class for storing node embeddings.

    The class is optimized for training large-scale node embeddings. It updates the embedding in
    a sparse way and can scale to graphs with millions of nodes. It also supports partitioning
    to multiple GPUs (on a single machine) for more acceleration. It does not support partitioning
    across machines.

    Currently, DGL provides two optimizers that work with this NodeEmbedding
    class: ``SparseAdagrad`` and ``SparseAdam``.

    NOTE: NodeEmbedding can only support single node single-GPU training and
    single node multi-GPU training.

    The implementation is based on torch.distributed package. It depends on the pytorch
    default distributed process group to collect multi-process information and uses
    ``torch.distributed.TCPStore`` to share meta-data information across multiple gpu processes.
    It use the local address of '127.0.0.1:12346' to initialize the TCPStore.

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
        global _store

        # Check whether it is multi-gpu training or not.
        if th.distributed.is_initialized():
            rank = th.distributed.get_rank()
            world_size = th.distributed.get_world_size()
        else:
            rank = -1
            world_size = 0
        self._rank = rank
        self._world_size = world_size
        host_name='127.0.0.1'
        port=12346

        if rank <= 0:
            emb = create_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)
            if init_func is not None:
                emb = init_func(emb)
        if rank == 0:
            if world_size > 1:
                # for multi-gpu training, setup a TCPStore for
                # embeding status synchronization across GPU processes
                if _store is None:
                    _store = th.distributed.TCPStore(
                        host_name, port, world_size, True, timedelta(seconds=30))
                for i in range(1, world_size):
                    # send embs
                    _store.set(name, name)
        elif rank > 0:
            # receive
            if _store is None:
                _store = th.distributed.TCPStore(
                    host_name, port, world_size, False, timedelta(seconds=30))
            _store.wait([name])
            emb = get_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)

        self._store = _store
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
        return self._store

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

    def set_optm_state(self, state):
        self._optm_state = state

    @property
    def optm_state(self):
        return self._optm_state

    @property
    def trace(self):
        return self._trace

    def reset_trace(self):
        self._trace = []

    @property
    def emb_tensor(self):
        return self._tensor
