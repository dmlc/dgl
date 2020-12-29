import tvm
from sddmm import sddmm_tree_reduction

binary_path = '../build/libfeatgraph_kernels.so'
kernels = []
idtypes = ['int32', 'int64']
dtypes = ['float16', 'float64', 'float32', 'int32', 'int64']

# SDDMM Tree Reduction
for dtype in dtypes:
    for idtype in idtypes:
        kernels.append(sddmm_tree_reduction(idtype, dtype))

module = tvm.build(kernels, target='cuda', target_host='llvm')
module.export_library(binary_path)
