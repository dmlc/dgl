import torch as th
from ...sparse import _gspmm, _gsddmm

def gspmm(g, op, reduce_op, lhs_data, rhs_data, rhs_data, lhs_target='u', rhs_target='e', out_target='v'):
    pass

def gsddmm(g, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    pass
