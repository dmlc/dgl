from .. import utils
from .. import ndarray
from .. import backend as F
from ..partition import NDArrayPartition

class MultiGPUTensor:
    def __init__(self, shape, dtype, device, comm, partition=None):
        """ Create a new Tensor stored across multiple GPUs according to
        `partition`. This funciton must be called by all processes.

        Parameters
        ----------
        shape : tuple
            The shape of the tensor. The tensor will be partitioned across its
            first dimension. As a result, dimensionless tensors are not
            allowed.
        device : 
            The current device.
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
        self._tensor = F.zeros(shape, dtype, device)

    def get_global(self, index):
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
            "tensor with one of same shape."
        self._tensor = F.copy_to(values, ctx=F.context(self._tensor))

    def update_local(self, index, values):
        """ Independently set rows of the local tensor to the given values.

        Parameters
        ----------
        index : Tensor
            The set of indices, in local space, to set on the current GPU.
        values : Tensor
            The set of values to set in the tensor stored on the current GPU.
        """
        self._tensor[index] = values

