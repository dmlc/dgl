import tvm
from sddmm_tr import sddmm_tree_reduction

binary_path = '../../build/libfeatgraph_kernels.so'
kernels = []
# SDDMM Tree Reduction
kernels.append(sddmm_tree_reduction('int32', 'float16'))
kernels.append(sddmm_tree_reduction('int32', 'float32'))
kernels.append(sddmm_tree_reduction('int32', 'float64'))
kernels.append(sddmm_tree_reduction('int64', 'float16'))
kernels.append(sddmm_tree_reduction('int64', 'float32'))
kernels.append(sddmm_tree_reduction('int64', 'float64'))

module = tvm.build(kernels, target='cuda', target_host='llvm')
module.export_library(binary_path)