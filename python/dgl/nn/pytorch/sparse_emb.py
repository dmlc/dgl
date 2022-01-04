"""Torch NodeEmbedding."""
from datetime import timedelta
import torch as th
from ...backend import pytorch as F
from ...utils import get_shared_mem_array, create_shared_mem_array
from ...cuda import nccl
from ...partition import NDArrayPartition

_STORE = None
_COMM = None

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
    device : th.device
        Device to store the embeddings on.
    parittion : NDArrayPartition
        The partition to use to distributed the embeddings between
        processes.

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
                 init_func=None, device=None, partition=None):
        global _STORE
        global _COMM

        if device is None:
            device = th.device('cpu')

        # Check whether it is multi-gpu training or not.
        if th.distributed.is_initialized():
            rank = th.distributed.get_rank()
            world_size = th.distributed.get_world_size()
        else:
            rank = -1
            world_size = 0
        self._rank = rank
        self._world_size = world_size
        self._store = None
        self._comm = None
        self._partition = partition

        host_name = '127.0.0.1'
        port = 12346

        if rank >= 0:
            # for multi-gpu training, setup a TCPStore for
            # embeding status synchronization across GPU processes
            if _STORE is None:
                _STORE = th.distributed.TCPStore(
                    host_name, port, world_size, rank == 0, timedelta(seconds=10*60))
            self._store = _STORE

        # embeddings is stored in CPU memory.
        if th.device(device) == th.device('cpu'):
            if rank <= 0:
                emb = create_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)
                if init_func is not None:
                    emb = init_func(emb)
            if rank == 0: # the master gpu process
                for _ in range(1, world_size):
                    # send embs
                    self._store.set(name, name)
            elif rank > 0:
                # receive
                self._store.wait([name])
                emb = get_shared_mem_array(name, (num_embeddings, embedding_dim), th.float32)
            self._tensor = emb
        else: # embeddings is stored in GPU memory.
            # setup nccl communicator
            if _COMM is None:
                if rank < 0:
                    _COMM = nccl.Communicator(1, 0, nccl.UniqueId())
                else:
                    # needs to be set for nccl to work
                    th.cuda.set_device(device)
                    if rank == 0:
                        # root process broadcasts nccl id
                        nccl_id = nccl.UniqueId()
                        self._store.set('nccl_root_id_sparse_emb', str(nccl_id))
                    else:
                        nccl_id = nccl.UniqueId(self._store.get('nccl_root_id_sparse_emb'))
                    _COMM = nccl.Communicator(self._world_size, self._rank,
                                              nccl_id)
            self._comm = _COMM

            if not self._partition:
                # for communication we need a partition
                self._partition = NDArrayPartition(
                    num_embeddings,
                    self._world_size if self._world_size > 0 else 1,
                    mode='remainder')

            # create local tensors for the weights
            local_size = self._partition.local_size(self._comm.rank())

            # TODO(dlasalle): support 16-bit/half embeddings
            emb = th.empty([local_size, embedding_dim], dtype=th.float32,
                           requires_grad=False, device=device)
            if init_func:
                emb = init_func(emb)
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
        if not self._comm or self._comm.size() == 1:
            emb = self._tensor[node_ids].to(device)
        else:
            if self.world_size > 0:
                emb = self._comm.sparse_all_to_all_pull(
                    node_ids, self._tensor, self._partition)
            else:
                emb = self._tensor[node_ids]
            emb = emb.to(device)
        if F.is_recording():
            emb = F.attach_grad(emb)
            self._trace.append((node_ids.to(device), emb))

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
    def comm(self):
        """Return dgl.cuda.nccl.Communicator for data
        sharing across processes.

        Returns
        -------
        dgl.cuda.nccl.Communicator
            Communicator used for data sharing.
        """
        return self._comm

    @property
    def partition(self):
        """Return the partition identifying how the tensor is split across
        processes.

        Returns
        -------
        String
            The mode.
        """

        return self._partition

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

    @property
    def embedding_dim(self):
        """Return the dimension of embeddings.

        Returns
        -------
        int
            The dimension of embeddings.
        """
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

        DEPRECATED: renamed weight

        Returns
        -------
        torch.Tensor
            The tensor storing the node embeddings
        """
        return self._tensor

    @property
    def weight(self):
        """Return the tensor storing the node embeddings

        Returns
        -------
        torch.Tensor
            The tensor storing the node embeddings
        """
        return self._tensor

    def all_set_embedding(self, values):
        """ Set the values of the embedding. This method must be called by all
        processes sharing the embedding with identical tensors for
        :attr:`values`.

        NOTE: This method must be called by all processes sharing the
        embedding, or it may result in a deadlock.

        Parameters
        ----------
        values : Tensor
            The global tensor to pull values from.
        """
        if self._partition:
            idxs = F.copy_to(
                self._partition.get_local_indices(
                    self._comm.rank(),
                    ctx=F.context(self._tensor)),
                F.context(values))
            self._tensor[:] = F.copy_to(F.gather_row(values, idxs),
                                        ctx=F.context(self._tensor))[:]
        else:
            if self._rank == 0:
                self._tensor[:] = F.copy_to(values,
                                            ctx=F.context(self._tensor))[:]
        if th.distributed.is_initialized():
            th.distributed.barrier()

    def all_get_embedding(self):
        """ Return a copy of the embedding stored in CPU memory. If this is a
        multi-processing instance, the tensor will be returned in shared
        memory. If the embedding is currently stored on multiple GPUs, all
        processes must call this method in the same order.

        NOTE: This method must be called by all processes sharing the
        embedding, or it may result in a deadlock.

        Returns
        -------
        torch.Tensor
            The tensor storing the node embeddings.
        """
        if self._partition:
            if self._world_size == 0:
                # non-multiprocessing
                return self._tensor.to(th.device('cpu'))
            else:
                # create a shared memory tensor
                shared_name = self._name + "_gather"
                if self._rank == 0:
                    # root process creates shared memory
                    emb = create_shared_mem_array(
                        shared_name,
                        (self._num_embeddings, self._embedding_dim),
                        self._tensor.dtype)
                    self._store.set(shared_name, shared_name)
                else:
                    self._store.wait([shared_name])
                    emb = get_shared_mem_array(
                        shared_name, (self._num_embeddings, self._embedding_dim),
                        self._tensor.dtype)
                # need to map indices and slice into existing tensor
                idxs = self._partition.map_to_global(
                    F.arange(0, self._tensor.shape[0],
                             ctx=F.context(self._tensor)),
                    self._rank).to(emb.device)
                emb[idxs] = self._tensor.to(emb.device)

                # wait for all processes to finish
                th.distributed.barrier()
                return emb
        else:
            # already stored in CPU memory
            return self._tensor
