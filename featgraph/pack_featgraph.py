""" Export featgraph kernels to a shared library. """
import tvm
from sddmm import sddmm_tree_reduction_gpu


def get_sddmm_kernels_gpu(idtypes, dtypes):
    """
    Parameters
    ----------
    idtypes: List[str]
        Possible index types.
    dtypes: List[str]
        Possible data types.

    Returns
    -------
    List[IRModule]:
        The list of IRModules.
    """
    ret = []
    # SDDMM Tree Reduction
    for dtype in dtypes:
        for idtype in idtypes:
            ret.append(sddmm_tree_reduction_gpu(idtype, dtype))

    return ret


if __name__ == '__main__':
    binary_path = 'libfeatgraph_kernels.so'
    kernels = []
    idtypes = ['int32', 'int64']
    dtypes = ['float16', 'float64', 'float32', 'int32', 'int64']

    kernels += get_sddmm_kernels_gpu(idtypes, dtypes)

    # build kernels and export the module to libfeatgraph_kernels.so
    module = tvm.build(kernels, target='cuda', target_host='llvm')
    module.export_library(binary_path)

