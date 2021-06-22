""" The MultiGPUTensor class. """

from .. import backend as F
from ..partition import NDArrayPartition

class MultiGPUTensor:
    """ Class for storing a large tensor split across GPU memory. When reading
    memory from otehr GPUs, the call must be from all GPUs storing the tensor.

    Example:
    --------

    It can be created via the shape and data type information from the tensor
    to split across gpus `large_cpu_tensor` in the below code:

    >>> split_tensor = MultiGPUTensor(large_cpu_tensor.shape,
                                      large_cpu_tensor.dtype,
                                      dev_id,
                                      nccl_comm)
    >>> split_tensor.set_global(large_cpu_tensor)

    Then, once it is stored across GPU memory, during training, feature for
    mini-batches can be fetched via the `get_global()` method. If we have the
    tensor of mini-batch nodes `input_nodes` we can use the following code to
    fetch the features for the mini-batch:
    ...
    >>> batch_features = split_tensor.get_global(input_nodes)
    """
    def __init__(self, shape, dtype, device, comm, partition=None):
        """ Create a new Tensor stored across multiple GPUs according to
        `partition`. This funciton must be called by all processes.

        Parameters
        ----------
        shape : tuple
            The shape of the tensor. The tensor will be partitioned across its
            first dimension. As a result, dimensionless tensors are not
            allowed.
        dtype : DType
            The backend data type.
        device : Device
            The current backend device.
        comm : nccl.Communicator
            The NCCL communicator to use.
        partition : NDArrayPartition, optional
            The partition describing how the tensor is split across the GPUs.
            If not specified, the indices will be striped evenly across the
            GPUs.
        """
        if partition is None:
            partition = NDArrayPartition(
                shape[0],
                comm.size(),
                mode='remainder')
        assert partition.num_parts() == comm.size(), "The partition " \
            "must have the same number of parts as the communicator has ranks."
        assert partition.array_size() == shape[0], "The partition must be for " \
            "an array with the same number of rows as this MultiGPUTensor."

        self._comm = comm
        self._partition = partition
        local_shape = list(shape)
        local_shape[0] = self._partition.local_size(self._comm.rank())
        self._tensor = F.zeros(local_shape, dtype, device)

    def all_gather_row(self, index):
        """ Synchronously with all other GPUs the tensor is stored on, gather
        the rows associated with the given set of indices on this GPU.
        This function must be called in all processes.

        Parameters
        ----------
        index : Tensor
            The set of indices, in global space, to fetch from across all GPUs.

        Returns
        -------
        Tensor
            The rows matching the set of requested indices.
        """
        return self._comm.sparse_all_to_all_pull(
            index, self._tensor, self._partition)

    def all_set_global(self, values):
        """ Set this process's portition of the global tensor. It will use the
        partition to select which rows of the global tensor should be stored in
        the current device.

        Parameters
        ----------
        values : Tensor
            The global tensor to pull values from.
        """
        idxs = F.copy_to(
            self._partition.get_local_indices(
                self._comm.rank(),
                ctx=F.context(self._tensor)),
            F.context(values))
        self.set_local(F.copy_to(F.gather_row(values, idxs),
                                 ctx=F.context(self._tensor)))

    def get_local(self):
        """ Independently get the local tensor of this GPU.

        Returns
        -------
        Tensor
            The current local tensor.
        """
        return self._tensor

    def set_local(self, values):
        """ Independently replace the content of the local tensor.

        Parameters
        ----------
        values : Tensor
            The tensor to replace the current one with. It must be of the same
            shape as this local tensor.
        """
        assert self._tensor.shape == values.shape, "Can only replace local " \
            "tensor with one of same shape: {} vs. {}".format(
                self._tensor.shape, values.shape)
        self._tensor = F.copy_to(values, ctx=F.context(self._tensor))
