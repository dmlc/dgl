"""For Tensor Serialization"""
from __future__ import absolute_import
from ..ndarray import NDArray
from .._ffi.function import _init_api
from .. import backend as F

__all__ = ['save_tensors', "load_tensors"]

_init_api("dgl.data.tensor_serialize")


def save_tensors(filename, tensor_dict):
    """
    Save dict of tensors to file
    
    Parameters
    ----------
    filename : str
        File name to store dict of tensors. 
    tensor_dict: dict of dgl NDArray or backend tensor
        Python dict using string as key and tensor as value

    Returns
    ----------
    status : bool
        Return whether save operation succeeds
    """
    nd_dict = {}
    is_empty_dict = len(tensor_dict) == 0
    for key, value in tensor_dict.items():
        if not isinstance(key, str):
            raise Exception("Dict key has to be str")
        if F.is_tensor(value):
            nd_dict[key] = F.zerocopy_to_dgl_ndarray(value)
        elif isinstance(value, NDArray):
            nd_dict[key] = value
        else:
            raise Exception(
                "Dict value has to be backend tensor or dgl ndarray")
    
    return _CAPI_SaveNDArrayDict(filename, nd_dict, is_empty_dict)


def load_tensors(filename, return_dgl_ndarray=False):
    """
    load dict of tensors from file
    
    Parameters
    ----------
    filename : str
        File name to load dict of tensors. 
    return_dgl_ndarray: bool
        Whether return dict of dgl NDArrays or backend tensors

    Returns
    ---------
    tensor_dict : dict
        dict of tensor or ndarray based on return_dgl_ndarray flag
    """
    nd_dict = _CAPI_LoadNDArrayDict(filename)
    tensor_dict = {}
    for key, value in nd_dict.items():
        if return_dgl_ndarray:
            tensor_dict[key] = value
        else:
            tensor_dict[key] = F.zerocopy_from_dgl_ndarray(value)
    return tensor_dict
