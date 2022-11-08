"""API creating NCCL communicators."""

from .. import backend as F
from .._ffi.function import _init_api

_COMM_MODES_MAP = {"remainder": 0}


class UniqueId(object):
    """Class for allowing python code to create and communicate NCCL Unique
    IDs, needed for creating communicators.
    """

    def __init__(self, id_str=None):
        """Create an object reference the current NCCL unique id."""
        if id_str:
            if isinstance(id_str, bytes):
                id_str = id_str.decode("utf-8")
            self._handle = _CAPI_DGLNCCLUniqueIdFromString(id_str)
        else:
            self._handle = _CAPI_DGLNCCLGetUniqueId()

    def get(self):
        """Get the C-handle for this object."""
        return self._handle

    def __str__(self):
        return _CAPI_DGLNCCLUniqueIdToString(self._handle)

    def __repr__(self):
        return "UniqueId[{}]".format(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Communicator(object):
    """High-level wrapper for NCCL communication."""

    def __init__(self, size, rank, unique_id):
        """Create a new NCCL communicator.

        Parameters
        ----------
        size : int
            The number of processes in the communicator.
        rank : int
            The rank of the current process in the communicator.
        unique_id : NCCLUniqueId
            The unique id of the root process (rank=0).

        Examples
        --------

        >>> from dgl.cuda.nccl import Communicator, UniqueId

        The root process will generate a unique NCCL id and communicate it
        to the other processes.

        >>> uid = UniqueId()
        >>> store.set('nccl_root_id', str(uid))

        And all other processes create unique ids from the root processes.

        >>> uid = UniqueId(store.get('nccl_root_id'))

        Then, all processes should create the communicator.

        >>> comm = Communicator(world_size, rank, uid)
        """
        assert rank < size, (
            "The rank of a process must be less than the "
            "size of the communicator."
        )
        self._handle = _CAPI_DGLNCCLCreateComm(size, rank, unique_id.get())
        self._rank = rank
        self._size = size

    def sparse_all_to_all_push(self, idx, value, partition):
        """Perform an all-to-all-v operation, where by all processors send out
        a set of indices and corresponding values. Indices and values,
        corresponding to the current process, will copied into the output
        arrays.

        Parameters
        ----------
        idx : tensor
            The 1D set of indices to send to other processors.
        value : tensor
            The multi-dimension set of values to send to other processors.
            The first dimension must match that of `idx`.
        partition : NDArrayPartition
            The object containing information for assigning indices to
            processors.

        Returns
        -------
        tensor
            The 1D tensor of the recieved indices.
        tensor
            The set of recieved values.

        Examples
        --------

        To perform a sparse_all_to_all_push(), a partition object must be
        provided. A partition of a homgeonous graph, where the vertices are
        striped across processes can be generated via:

        >>> from dgl.partition import NDArrayPartition
        >>> part = NDArrayPartition(g.num_nodes(), comm.size(), mode='remainder' )

        With this partition, each processor can send values to be associatd
        with vertices in the graph. So if we have an array `global_idxs` of all of
        the neighbors updated during mini-batch processing, and an array
        `global_values` containing the new values associated with the neighbors,
        we communicate them to the own processes via:

        >>> my_idxs, my_values = comm.sparse_all_to_all_push(global_idxs, global_values, part)

        This communication pattern is common when communicating gradient
        updates for node embeddings.

        Indices the current process owns, do not need to treated specially,
        as internally they will be copied to the output array. If we have a
        set of indices in process 0 '[0, 3, 8, 9, 10]` and for process 1
        '[0, 2, 4, 5, 8, 8, 9]'. Using a remainder partition will result
        indices for processe 0 of '[0, 8, 10, 0, 2, 4, 8, 8]', and for
        process 1 of '[3, 9, 5, 9]'.
        """
        out_idx, out_value = _CAPI_DGLNCCLSparseAllToAllPush(
            self.get(),
            F.zerocopy_to_dgl_ndarray(idx),
            F.zerocopy_to_dgl_ndarray(value),
            partition.get(),
        )
        return (
            F.zerocopy_from_dgl_ndarray(out_idx),
            F.zerocopy_from_dgl_ndarray(out_value),
        )

    def sparse_all_to_all_pull(self, req_idx, value, partition):
        """Perform an all-to-all-v operation, where by all processors request
        the values corresponding to their set of indices.

        Parameters
        ----------
        req_idx : IdArray
            The set of indices this processor is requesting.
        value : NDArray
            The multi-dimension set of values that can be requested from
            this processor.
        partition : NDArrayPartition
            The object containing information for assigning indices to
            processors.

        Returns
        -------
        tensor
            The set of recieved values, corresponding to `req_idx`.

        Examples
        --------

        To perform a sparse_all_to_all_pull(), a partition object must be
        provided. A partition of a homgeonous graph, where the vertices are
        striped across processes can be generated via:

        >>> from dgl.partition import NDArrayPartition
        >>> part = NDArrayPartition(g.num_nodes(), comm.size(), mode='remainder' )

        With this partition, each processor can request values/features
        associated with vertices in the graph. So in the case where we have
        a set of neighbors 'nbr_idxs' we need features for, and each process
        has a tensor 'node_feat' storing the features of nodes it owns in
        the partition, the features can be requested via:

        >>> nbr_values = comm.sparse_all_to_all_pull(nbr_idxs, node_feat, part)

        Then two the arrays 'nbr_idxs' and 'nbr_values' forms the sparse
        set of features, where 'nbr_idxs[i]' is the global node id, and
        'nbr_values[i]' is the feature vector for that node. This
        communication pattern is useful for node features or node
        embeddings.
        """
        out_value = _CAPI_DGLNCCLSparseAllToAllPull(
            self.get(),
            F.zerocopy_to_dgl_ndarray(req_idx),
            F.zerocopy_to_dgl_ndarray(value),
            partition.get(),
        )
        return F.zerocopy_from_dgl_ndarray(out_value)

    def get(self):
        """Get the C-Handle for this object."""
        return self._handle

    def rank(self):
        """Get the rank of this process in this communicator.

        Returns
        -------
        int
            The rank of this process.
        """
        return self._rank

    def size(self):
        """Get the size of this communicator.

        Returns
        -------
        int
            The number of processes in this communicator.
        """
        return self._size


def is_supported():
    """Check if DGL was built with NCCL support.

    Returns
    -------
    bool
        True if NCCL support was built in.
    """
    return _CAPI_DGLNCCLHasSupport()


_init_api("dgl.cuda.nccl")
