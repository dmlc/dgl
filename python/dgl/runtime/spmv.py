"""Module for SPMV rules."""
from __future__ import absolute_import

from ..base import DGLError
from . import ir

def analyze_v2v_spmv(graph, mfunc, rfunc):
    """Analyze if SPMV from node space to node space can be applied

    Parameters
    ----------
    graph: DGLGraph
        DGLGraph to use
    mfunc : list of dgl.function.BuiltinFunction
        The message function list.
    rfunc : list of dgl.function.BuiltinFunction
        The reduce function list.

    Returns
    -------
    spmv_pairs : list of pair of builtin functions
        The pair of spvm applicable message/reduce functions.
    mfunc_left: list
        A list of message functions that can't use v2v spmv. In other
        words, these message functions need to be materialized
    rfunc_left: list
        A list of reduce functions that can't use v2v spmv
    """
    spmv_pairs = []
    mfunc_left = []
    rfunc_left = []

    fld2mfunc = {fn.out_field: fn for fn in mfunc}
    touched_mfld = set()

    for rfn in rfunc:
        mfld = rfn.msg_field
        if mfld not in fld2mfunc:
            raise DGLError('Reduce function requires message field "%s",'
                           ' but no message function generates it.' % mfld)
        mfn = fld2mfunc[mfld]
        # FIXME: should pre-compile a look up table
        if mfn.is_spmv_supported(graph) and rfn.is_spmv_supported():
            spmv_pairs.append((mfn, rfn))
        else:
            if mfld not in touched_mfld:
                touched_mfld.add(mfld)
                mfunc_left.append(mfn)
            rfunc_left.append(rfn)

    return spmv_pairs, mfunc_left, rfunc_left

def analyze_e2v_spmv(graph, rfunc):
    """Analyze if SPMV from edge space to node space can be applied

    Parameters
    ----------
    graph: DGLGraph
        DGLGraph to use
    rfunc : list of dgl.function.BuiltinFunction
        The reduce function list.

    Returns
    -------
    spmv_rfunc : list
        A list of spmv-applicable reduce builtins.
    rfunc_left : list
        A list of reduce builtins that are not applicable
    """
    spmv_rfunc = []
    rfunc_left = []
    for rfn in rfunc:
        if rfn.is_spmv_supported():
            spmv_rfunc.append(rfn)
        else:
            rfunc_left.append(rfn)
    return spmv_rfunc, rfunc_left

def gen_v2v_spmv_schedule_uv(spmv_pairs, nf, ef, u, v, eid):
    """
    spmv_pairs : (MessageFunction, ReduceFunction)
    nf : ir.Var
    ef : ir.Var
    u : ir.Var
    v : ir.Var
    eid : ir.Var
    """
    adj_idx = None
    adj = None
    fld_list = []
    ft_list = []
    for mfn, rfn in spmv_pairs:
        print('v2v mfn=%s rfn=%s' % (mfn.name, rfn.name))
        if mfn.use_edge_feature:
            if adj_idx is None:
                adj_idx = ir.Var.SPMAT(_build_adj_matrix_index_uv(u, v))
            ftedge = ir.READ(ef, eid, ir.Var.STR(mfn.edge_field))
            ftsrc = ir.READ_COL(nf, ir.Var.STR(mfn.src_field))
            ftdst = ir.SPMV_WITH_DATA(adj_idx, ftedge, ftsrc)
        else:
            if adj is None:
                adj = ir.Var.SPMAT(_build_adj_matrix_uv(u, v))
            ftsrc = ir.READ_COL(nf, ir.Var.STR(mfn.src_field))
            ftdst = ir.SPMV(adj, ftsrc)
        # save for merge
        fld_list.append(ir.Var.STR(mfn.out_field))
        ft_list.append(ftdst)
    # TODO: MERGE_COL
    reduced_feat = ir.MERGE(fld_list, ft_list)
    # TODO: v
    return v, reduced_feat

def _build_adj_matrix_index_uv(u, v):
    return None

def _build_adj_matrix_uv(u, v):
    return None
