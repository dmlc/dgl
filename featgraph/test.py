import torch
import tvm
from operators import gsddmm, gspmm, segment_reduce, segment_gemm


def test_codegen():
    #print(gsddmm('add', 3, 'int32', 'float32', schedule_type='general'))
    #print(gsddmm('dot', 2, 'int32', 'float32', schedule_type='tree'))
    #print(tvm.build(gspmm('copy_rhs', 'sum', 1, 'int32', 'float32', schedule_type='general'), target='cuda', target_host='llvm').imported_modules[0].get_source())
    #print(tvm.build(segment_reduce('sum', 'int32', 'float32', schedule_type='tree'), target='cuda', target_host='llvm').imported_modules[0].get_source())
    #print(tvm.build(gspmm('copy_lhs', 'sum', 1, 'int32', 'float32', schedule_type='merge'), target='cuda', target_host='llvm').imported_modules[0].get_source())
    #print(tvm.build(segment_gemm('int32', 'float32'), target='cuda', target_host='llvm').imported_modules[0].get_source())
    print(segment_gemm('int32', 'float16'))

if __name__ == "__main__":
    test_codegen()

