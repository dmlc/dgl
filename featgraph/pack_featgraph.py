""" Export featgraph kernels to a shared library. """
import tvm
from gsddmm import gsddmm
from utils import irreducible_ops


def get_sddmm_kernels_gpu(binary_ops, idtypes, dtypes, schedule_types):
    """
    Parameters
    ----------
    binary_ops: List[str]
        Possible binary operators.
    idtypes: List[str]
        Possible index types.
    dtypes: List[str]
        Possible data types.
    schedules_types: List[str] 
        Possbiel schedule types.

    Returns
    -------
    List[IRModule]:
        The list of IRModules.
    """
    ret = []
    # SDDMM Tree Reduction
    for binary_op in binary_ops:
        for dtype in dtypes:
            for idtype in idtypes:
                for sche_type in schedule_types:
                    if binary_op != 'dot' and sche_type == 'tree':
                        continue
                    for ndim in range(1, 5):
                        ret.append(gsddmm(binary_op, ndim, idtype, dtype, lhs_target=0, rhs_target=1, schedule_type=sche_type))
                        ret.append(gsddmm(binary_op, ndim, idtype, dtype, lhs_target=0, rhs_target=2, schedule_type=sche_type))
                        ret.append(gsddmm(binary_op, ndim, idtype, dtype, lhs_target=1, rhs_target=2, schedule_type=sche_type))

    return ret


if __name__ == '__main__':
    binary_path = 'libfeatgraph_kernels.so'
    kernels = []
    idtypes = ['int32', 'int64']
    dtypes =  ['float16', 'float64', 'float32']#, 'int32', 'int64']
    binary_ops = ['add', 'mul', 'dot']
    schedule_types = ['general', 'tree']

    kernels += get_sddmm_kernels_gpu(binary_ops, idtypes, dtypes, schedule_types)

    # build kernels and export the module to libfeatgraph_kernels.so
    module = tvm.build(kernels, target='cuda', target_host='llvm')
    module.export_library(binary_path)

