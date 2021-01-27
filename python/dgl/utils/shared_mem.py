
"""Shared memory utilities."""
from .. import backend as F
from .._ffi.ndarray import empty_shared_mem

def get_shared_mem_array(name, shape, dtype):
    """ Get a tensor from shared memory with specific name

    Parameters
    ----------
    name : str
        The unique name of the shared memory
    shape : tuple of int
        The shape of the returned tensor
    dtype : F.dtype
        The dtype of the returned tensor

    Returns
    -------
    F.tensor
        The tensor got from shared memory.
    """
    name = 'DGL_'+name
    new_arr = empty_shared_mem(name, False, shape, F.reverse_data_type_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def create_shared_mem_array(name, shape, dtype):
    """ Create a tensor from shared memory with the specific name

    Parameters
    ----------
    name : str
        The unique name of the shared memory
    shape : tuple of int
        The shape of the returned tensor
    dtype : F.dtype
        The dtype of the returned tensor

    Returns
    -------
    F.tensor
        The created tensor.
    """
    name = 'DGL_'+name
    new_arr = empty_shared_mem(name, True, shape, F.reverse_data_type_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)
