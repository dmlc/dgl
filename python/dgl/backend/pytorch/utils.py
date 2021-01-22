from .tensor import data_type_dict, zerocopy_from_dlpack
from ..._ffi.ndarray import empty_shared_mem

dtype_dict = data_type_dict()
dtype_dict = {dtype_dict[key]:key for key in dtype_dict}

def get_shared_mem_array(name, shape, dtype):
    name = 'DGL_'+name
    new_arr = empty_shared_mem(name, False, shape, dtype_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return zerocopy_from_dlpack(dlpack)

def create_shared_mem_array(name, shape, dtype):
    name = 'DGL_'+name
    new_arr = empty_shared_mem(name, True, shape, dtype_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return zerocopy_from_dlpack(dlpack)