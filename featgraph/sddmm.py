""" The compute function and schedules for SDDMM kernels written in TVM. """
import tvm
from tvm import te


def sddmm_tree_reduction_gpu(idx_type, feat_type):
    """ SDDMM kernels on GPU optimized with Tree Reduction.
    
    Parameters
    ----------
    idx_type : str
        The data type for indexing tensors.
    feat_type : str
        The data type of feature tensor.

    Returns
    -------
    IRModule
        The result IRModule.
    """
    # define vars and placeholders
    nnz = te.var('nnz', idx_type)
    num_rows = te.var('num_rows', idx_type)
    num_cols = te.var('num_cols', idx_type)
    H = te.var('num_heads', idx_type)
    D = te.var('feat_len', idx_type)
    row = te.placeholder((nnz,), idx_type, 'row')
    col = te.placeholder((nnz,), idx_type, 'col')
    ufeat = te.placeholder((num_rows, H, D), feat_type, 'ufeat')
    vfeat = te.placeholder((num_cols, H, D), feat_type, 'vfeat')
    # define edge computation function
    def edge_func(eid, h, i):
        k = te.reduce_axis((0, D), name='k')
        return te.sum(ufeat[row[eid], h, k] * vfeat[col[eid], h, k], axis=k)
    out = te.compute((nnz, H, tvm.tir.IntImm(idx_type, 1)), edge_func, name='out')
    # define schedules
    sched = te.create_schedule(out.op)
    edge_axis, head_axis, _ = out.op.axis
    reduce_axis = out.op.reduce_axis[0]
    _, red_inner = sched[out].split(reduce_axis, factor=32)
    edge_outer, edge_inner = sched[out].split(edge_axis, factor=32)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
    sched[out].bind(edge_outer, te.thread_axis('blockIdx.x'))
    sched[out].bind(head_axis, te.thread_axis('blockIdx.y'))
    return tvm.lower(sched, [row, col, ufeat, vfeat, out],
                     name='SDDMMTreeReduction_{}_{}'.format(idx_type, feat_type))


if __name__ == '__main__':
    kernel0 = sddmm_tree_reduction_gpu('int32', 'float32')
    print(kernel0)

