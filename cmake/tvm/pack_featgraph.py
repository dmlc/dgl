import os
from sddmm_tr import sddmm_tree_reduction

lib_path = '../../build/libfeatgraph.so'

f = sddmm_tree_reduction('int32', 'float32')
f.export_library(lib_path)