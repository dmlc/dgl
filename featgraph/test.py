import torch
import tvm
from operators import gsddmm, gspmm, segment_reduce


def test_gsddmm():
    #print(gsddmm('add', 3, 'int32', 'float32', schedule_type='general'))
    #print(gsddmm('dot', 2, 'int32', 'float32', schedule_type='tree'))
    #print(tvm.build(gspmm('copy_rhs', 'max', 1, 'int32', 'float32'), target='cuda', target_host='llvm').imported_modules[0].get_source())
    print(segment_reduce('sum', 'int32', 'float32'))
    pass

if __name__ == "__main__":
    test_gsddmm()
