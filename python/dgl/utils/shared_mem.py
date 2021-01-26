
from .. import backend as F
from .._ffi.ndarray import empty_shared_mem

def get_shared_mem_array(name, shape, dtype):
    name = 'DGL_'+name
    new_arr = empty_shared_mem(name, False, shape, F.reverse_data_type_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def create_shared_mem_array(name, shape, dtype):
    name = 'DGL_'+name
    new_arr = empty_shared_mem(name, True, shape, F.reverse_data_type_dict[dtype])
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)