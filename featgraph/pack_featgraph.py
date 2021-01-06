""" Export featgraph kernels to a shared library. """
import tvm
from operators import gsddmm


def get_sddmm_kernels(binary_ops, idtypes, dtypes, schedule_types, target):
    """
    Parameters
    ----------
    binary_ops : List[str]
        Possible binary operators.
    idtypes : List[str]
        Possible index types.
    dtypes : List[str]
        Possible data types.
    schedules_types : List[str] 
        Possbiel schedule types.
    target : str
        Could be ``llvm`` or ``cuda``.

    Returns
    -------
    List[IRModule]:
        The list of IRModules.
    """
    ret = []
    # SDDMM Tree Reduction
    for binary_op in binary_ops:
        for sche_type in schedule_types:
            if binary_op != 'dot' and sche_type == 'tree':
                continue
            for lhs_target in ['u', 'e']:
                for rhs_target in ['e', 'v']:
                    if lhs_target == rhs_target:
                        continue
                    for ndim in range(1, 5):
                        for dtype in dtypes:
                            for idtype in idtypes:
                                ret.append(
                                    gsddmm(binary_op, ndim, idtype, dtype,
                                           lhs_target=lhs_target,
                                           rhs_target=rhs_target,
                                           schedule_type=sche_type,
                                           target=target))
    return ret


if __name__ == '__main__':
    binary_path = 'libfeatgraph_kernels.so'
    kernels = []
    idtypes = ['int32', 'int64']
    dtypes =  ['float16', 'float64', 'float32']#, 'int32', 'int64']
    binary_ops = ['add', 'mul', 'dot']
    schedule_types = ['general', 'tree']
    kernels += get_sddmm_kernels(binary_ops, idtypes, dtypes, schedule_types, 'cuda')
    # build kernels and export the module to libfeatgraph_kernels.so
    module = tvm.build(kernels, target='cuda', target_host='llvm')
    module.export_library(binary_path)
