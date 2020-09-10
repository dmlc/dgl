import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm

from ..base import DGLError
from .util import binary_op_map, reduce_op_map

def _spmm(out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
          ufeat, efeat, edge_id, pack, num_feat_partitions):
    reshapes = []
    if pack:
        nnz = efeat.shape[0]
        feat_len = topi.util.get_const_int(topi.util.prod(efeat.shape[1:]))
        feat_len_per_partition = feat_len // num_feat_partitions
        def reshape(feat_outer, n, feat_inner):
            fid = feat_outer * feat_len_per_partition + feat_inner
            return efeat.__getitem__((n,) + tuple(topi.util.unravel_index(fid, efeat.shape[1:])))
        reshaped_efeat = te.compute((num_feat_partitions, nnz, feat_len_per_partition),
                                    reshape, name='reshaped_efeat')
        reshapes.append(reshaped_efeat)
    def msgfunc(*args):
        row = args[0]
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        elem_idx = te.reduce_axis((0, row_end-row_start), name="elem_idx")
        u_val = ufeat.__getitem__((adj_indices[row_start + elem_idx],) + args[1:])
        if pack:
            fid = topi.util.ravel_index(args[1:], out_shp[1:])
            e_val = reshaped_efeat[fid // feat_len_per_partition, row_start + elem_idx,
                                   fid % feat_len_per_partition]
        else:
            e_val = efeat.__getitem__((edge_id(row_start + elem_idx),) + args[1:])
        if reduce_op == 'sum':
            return te.sum(binary_op_map[binary_op](u_val, e_val), axis=elem_idx)
        else:
            if binary_op == 'copy_lhs':
                return reduce_op_map[reduce_op]((adj_indices[row_start + elem_idx], u_val),
                                                axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), e_val),
                                                axis=elem_idx)
            else:
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx),
                                                 adj_indices[row_start + elem_idx],
                                                 binary_op_map[binary_op](u_val, e_val)),
                                                axis=elem_idx)
    rst = te.compute(out_shp, msgfunc, name='out')
    if reduce_op == 'sum':
        rst = (rst,)
    return rst, reshapes

def _spmm_dds(out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
              ufeat, efeat, edge_id, pack, num_feat_partitions):
    num_col_partitions = adj_indptr.shape[0]
    reshapes = []
    if pack:
        #same as above
        nnz = efeat.shape[0]
        feat_len = topi.util.get_const_int(topi.util.prod(ufeat.shape[1:]))
        # assume divide num_feat_partitions
        feat_len_per_partition = feat_len // num_feat_partitions
        def reshape(feat_outer, n, feat_inner):
            fid = feat_outer * feat_len_per_partition + feat_inner
            return efeat.__getitem__((n,) + tuple(topi.util.unravel_index(fid, efeat.shape[1:])))
        reshaped_efeat = te.compute((num_feat_partitions, nnz, feat_len_per_partition),
                                    reshape, name='reshaped_efeat')
        reshapes.append(reshaped_efeat)
    def msgfunc(*args):
        col_part_idx = args[0]
        row = args[1]
        row_start = adj_indptr[col_part_idx, row]
        row_end = adj_indptr[col_part_idx, row + 1]
        elem_idx = te.reduce_axis((0, row_end-row_start), name="elem_idx")
        u_val = ufeat.__getitem__((adj_indices[row_start + elem_idx],) + args[2:])
        if pack:
            fid = topi.util.ravel_index(args[2:], out_shp[1:])
            e_val = reshaped_efeat[fid // feat_len_per_partition,
                                   row_start + elem_idx,
                                   fid % feat_len_per_partition]
        else:
            e_val = efeat.__getitem__((edge_id(row_start + elem_idx),) + args[2:])
        if reduce_op == 'sum':
            return te.sum(binary_op_map[binary_op](u_val, e_val), axis=elem_idx)
        else:
            if binary_op == 'copy_lhs':
                return reduce_op_map[reduce_op]((adj_indices[row_start + elem_idx], u_val),
                                                axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), e_val),
                                                axis=elem_idx)
            else:
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx),
                                                 adj_indices[row_start + elem_idx],
                                                 binary_op_map[binary_op](u_val, e_val)),
                                                axis=elem_idx)
    # process segments
    if reduce_op == 'sum':
        intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc,
                                  name='intermediate')
    else:
        if binary_op == 'copy_lhs':
            argu, intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc,
                                            name='intermediate')
        elif binary_op == 'copy_rhs':
            arge, intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc,
                                            name='intermediate')
        else:
            arge, argu, intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc,
                                                  name='intermediate')
    k = te.reduce_axis((0, num_col_partitions), name='k')
    # merge segments
    if reduce_op == 'sum':
        rst = (te.compute(out_shp, lambda *args: te.sum(intermediate.__getitem__((k,) + args),
                                                        axis=k), name='out'), )
    else:
        if binary_op == 'copy_lhs':
            rst = te.compute(out_shp, lambda *args: reduce_op_map[reduce_op](
                (argu.__getitem__((k,) + args), intermediate.__getitem__((k,) + args)), axis=k),
                             name='out')
        elif binary_op == 'copy_rhs':
            rst = te.compute(out_shp, lambda *args: reduce_op_map[reduce_op](
                (arge.__getitem__((k,) + args), intermediate.__getitem__((k,) + args)), axis=k),
                             name='out')
        else:
            rst = te.compute(out_shp, lambda *args: reduce_op_map[reduce_op](
                (arge.__getitem__((k,) + args), argu.__getitem__((k,) + args),
                 intermediate.__getitem__((k,) + args)), axis=k),
                             name='out')
    return rst, intermediate, reshapes

def _spmm_cuda_general(sched, out):
    node_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    sched[out].bind(node_axis, te.thread_axis('blockIdx.x'))
    sched[out].bind(feat_axis, te.thread_axis('threadIdx.x'))

def _spmm_cuda_tree_reduce(sched, out):
    reduce_axis = out.op.reduce_axis[0]
    node_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    _, red_inner = sched[out].split(reduce_axis, factor=32)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(feat_axis, te.thread_axis('threadIdx.y'))
    sched[out].bind(node_axis, te.thread_axis('blockIdx.x'))

def _spmm_cpu(sched, out, num_feat_partitions, reshapes):
    reduce_axis = out.op.reduce_axis[0]
    node_axis = out.op.axis[0]
    if num_feat_partitions == 1:
        sched[out].reorder(node_axis, reduce_axis, *out.op.axis[1:])
    else:
        feat_axis = sched[out].fuse(*out.op.axis[1:])
        feat_outer, feat_inner = sched[out].split(feat_axis, nparts=num_feat_partitions)
        sched[out].reorder(feat_outer, node_axis, reduce_axis, feat_inner)
        for reshape in reshapes:
            sched[reshape].compute_at(sched[out], feat_outer)
            sched[reshape].parallel(reshape.op.axis[1])
    sched[out].parallel(node_axis)

def _spmm_dds_sched(sched, out, intermediate, num_feat_partitions, reshapes):
    ifeat_axis = sched[intermediate].fuse(*intermediate.op.axis[2:])
    inter_col_part_axis = intermediate.op.axis[0]
    inter_row_axis = intermediate.op.axis[1]
    inter_red_axis = intermediate.op.reduce_axis[0]
    sched[intermediate].reorder(inter_col_part_axis, inter_row_axis,
                                inter_red_axis, ifeat_axis)
    ofeat_axis = sched[out].fuse(*out.op.axis[1:])
    if num_feat_partitions == 1:
        sched[out].reorder(out.op.reduce_axis[0], out.op.axis[0], ofeat_axis)
    else:
        # only need to partition feature axis of out
        ofo, ofi = sched[out].split(ofeat_axis, nparts=num_feat_partitions)
        sched[out].reorder(ofo, out.op.reduce_axis[0], out.op.axis[0], ofi)
        for reshape in reshapes:
            sched[reshape].compute_at(sched[out], ofo)
            sched[reshape].parallel(reshape.op.axis[1])
    sched[intermediate].compute_at(sched[out], out.op.reduce_axis[0])
    sched[intermediate].parallel(inter_row_axis)
    sched[out].parallel(out.op.axis[0])

def spmm(binary_op, reduce_op, nnz, num_rows, num_cols,
         lhs_shp, rhs_shp, out_shp,
         indice_type, feat_type, use_idx=False,
         num_col_partitions=1, num_feat_partitions=1, target='llvm'):
    if target == 'cuda' and (num_col_partitions > 1 or num_feat_partitions > 1):
        raise DGLError('Cannot build cuda kernel with partition')
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    else:
        raise DGLError('Unrecognized number of bits')
    if '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    elif '16' in feat_type:
        feat_type = 'float16'
    else:
        raise DGLError('Unrecognized number of bits')
    # check if used for sampling
    generic_shape = nnz == 0 and num_rows == 0 and num_cols == 0
    if generic_shape:
        num_rows = te.var('num_rows', indice_type)
        num_cols = te.var('num_cols', indice_type)
        nnz = te.var('nnz', indice_type)
        if num_col_partitions > 1:
            raise DGLError('Column Partitioning with sampling not supported')
    else:
        # convert python int into tir nodes so that type of iterator in generated code is correct
        num_rows = IntImm(indice_type, num_rows)
        num_cols = IntImm(indice_type, num_cols)
        nnz = IntImm(indice_type, nnz)
    # check if use broadcast
    use_bcast = (binary_op not in ['copy_lhs', 'copy_rhs']) and lhs_shp != rhs_shp
    # placeholder for sparse matrix
    if num_col_partitions > 1:
        adj_indptr = te.placeholder((num_col_partitions, num_rows+1), indice_type, 'adj_indptr')
    else:
        adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    efeat = te.placeholder((nnz,) + tuple([IntImm(indice_type, s) for s in rhs_shp]),
                           feat_type, 'efeat')
    edge_mapping = te.placeholder((nnz,), indice_type, 'edge_mapping')
    # placeholder for dense features
    ufeat = te.placeholder((num_cols,) + tuple([IntImm(indice_type, s) for s in lhs_shp]),
                           feat_type, 'ufeat')
    use_u = binary_op != 'copy_rhs'
    use_e = binary_op != 'copy_lhs'
    def edge_id(x):
        return edge_mapping[x] if use_idx else x
    # if edge_mapping is present, do not use array packing
    # cannot use pack with bcast
    use_pack = num_feat_partitions > 1 and not use_bcast and use_e and not use_idx
    # compute
    out_shp = (num_rows,) + tuple([IntImm(indice_type, s) for s in out_shp])
    if num_col_partitions == 1:
        rst, reshapes = _spmm(out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
                              ufeat, efeat, edge_id, use_pack, num_feat_partitions)
    else:
        # same
        rst, intermediate, reshapes = _spmm_dds(out_shp, binary_op, reduce_op,
                                                adj_indptr, adj_indices,
                                                ufeat, efeat, edge_id, use_pack,
                                                num_feat_partitions)
    out = rst[-1]
    # schedule
    sched = te.create_schedule(out.op)
    if target == 'cuda':
        # cuda schedule
        if topi.util.get_const_int(topi.util.prod(out.shape[1:])) < 16:
            # use tree reduce if feat_len is small
            _spmm_cuda_tree_reduce(sched, out)
        else:
            #othrewise just parallel on feature dimension
            _spmm_cuda_general(sched, out)
    else:
        # llvm schedule
        if num_col_partitions == 1:
            _spmm_cpu(sched, out, num_feat_partitions, reshapes)
        else:
            _spmm_dds_sched(sched, out, intermediate, num_feat_partitions, reshapes)
        # prepare input
    f_input = [adj_indptr, adj_indices]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, indice_type, feat_type
        ])
    if use_idx:
        f_input.append(edge_mapping)
        f_name += '_idx'
    if use_u:
        f_input.append(ufeat)
    if use_e:
        f_input.append(efeat)
    f_input += rst
    if generic_shape:
        f_input += [num_rows, num_cols, nnz]
    # bind autobroadcast buffer
    u_buffer = tvm.tir.decl_buffer(ufeat.shape, ufeat.dtype, name='u_buf',
                                   buffer_type='auto_broadcast')
    e_buffer = tvm.tir.decl_buffer(efeat.shape, efeat.dtype, name='e_buf',
                                   buffer_type='auto_broadcast')
    binds = {}
    if use_bcast:
        binds = {ufeat:u_buffer, efeat: e_buffer}
    # print(tvm.lower(sched, f_input, binds=binds))
    return tvm.build(sched, f_input, target=target, name=f_name, binds=binds)
    