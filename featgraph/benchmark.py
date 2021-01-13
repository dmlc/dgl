import torch
import tvm
import time
import numpy as np
from operators import gsddmm, gspmm


def bench_workload(workload):
    """Benchmark a workload
    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats

def reordering():
    pass


def bench_spmm():
    mod = tvm.build(
        gspmm('copy_lhs', 'sum', 1, 'int32', 'float32', schedule_type='merge'),
        target='cuda',
        target_host='llvm'
    )
    
    ctx = tvm.context('cuda', 0)
    for dataset in ['reddit', 'arxiv', 'proteins']:
        with open('dataset/{}_csr.npy'.format(dataset), 'rb') as f:
            indptr_np = np.load(f)
            indices_np = np.load(f)

        indptr = tvm.nd.array(indptr_np, ctx)
        indices = tvm.nd.array(indices_np, ctx)
        n = len(indptr_np) - 1
        ufeat = tvm.nd.array(np.random.uniform(size=(n, 128)).astype('float32'), ctx)
        out = tvm.nd.array(np.zeros(shape=(n, 128)).astype("float32"), ctx)

        mod(indptr, indices, ufeat, out)
        def workload(nrepeats):
            timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
            return timer(indptr, indices, ufeat, out).mean * nrepeats

        print(bench_workload(workload))
    

def bench_sddmm():
    # TODO(zihao)
    pass


if __name__ == "__main__":
    bench_spmm()
    bench_sddmm()
