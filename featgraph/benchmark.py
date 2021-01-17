from scipy.sparse.coo import coo_matrix
import torch
import tvm
import time
import numpy as np
from tvm._ffi.runtime_ctypes import DataType
from operators import gsddmm, gspmm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def reordering(indptr, indices, row, col):
    data = np.ones_like(indices)
    n = len(indptr) - 1
    graph = csr_matrix((data, indices, indptr), shape=(n, n))
    perm = reverse_cuthill_mckee(graph)
    row_new = perm.take(row)
    col_new = perm.take(col)
    graph_new = csr_matrix((data, (col_new, row_new)), shape=(n, n))
    return graph_new.indptr, graph_new.indices


def bench_spmm():
    mod = tvm.build(
        gspmm('copy_lhs', 'min', 'int32', 'float16', schedule_type='general'),
        target='cuda',
        target_host='llvm'
    )
    print(mod.imported_modules[0].get_source())
    
    ctx = tvm.context('cuda', 0)
    for dataset in ['reddit', 'arxiv', 'proteins']:
        with open('dataset/{}_csr.npy'.format(dataset), 'rb') as f:
            indptr_np = np.load(f)
            indices_np = np.load(f)
        with open('dataset/{}_coo.npy'.format(dataset), 'rb') as f:
            row_np = np.load(f)
            col_np = np.load(f)

        # w/o reordering
        indptr = tvm.nd.array(indptr_np, ctx)
        indices = tvm.nd.array(indices_np, ctx)
        n = len(indptr_np) - 1
        ufeat = tvm.nd.array(np.random.uniform(size=(n, 128)).astype('float32'), ctx)
        out = tvm.nd.array(np.zeros(shape=(n, 128)).astype("float32"), ctx)

        def bench_workload(*args):
            timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=10)
            return timer(*args).mean

        print(bench_workload(indptr, indices, ufeat, out))

        # w/ reordering
        """
        indptr_np_1, indices_np_1 = reordering(indptr_np, indices_np, row_np, col_np)
        indptr_1 = tvm.nd.array(indptr_np_1, ctx)
        indices_1 = tvm.nd.array(indices_np_1, ctx)

        print(bench_workload(get_workload(indptr_1, indices_1, ufeat, out)))
        """


def bench_sddmm():
    # TODO(zihao)
    pass


if __name__ == "__main__":
    bench_spmm()
    bench_sddmm()
