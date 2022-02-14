"""dgl gather_mm operator module."""
from ..backend import gather_mm as gather_mm_internal
from ..backend import segment_mm as segment_mm_internal

__all__ = ['gather_mm', 'segment_mm']

def segment_mm(lhs_data, rhs_data, seglen_lhs):
    r""" Performs matrix multiplication according to segments.
    Suppose ``seglen_lhs == [10, 5, 0, 3]``, the operator will perform
    four matrix multiplications:
    lhs_data[0:10] @ rhs_data[0], lhs_data[10:15] @ rhs_data[1],
    lhs_data[15:15] @ rhs_data[2], lhs_data[15:18] @ rhs_data[3]

    Parameters
    ----------
    lhs_data : tensor
        The left operand, 2-D tensor of shape (N, D1)
    rhs_data : tensor
        The right operand, 2-D tensor of shape (R * D1, D2)
    seglen_lhs : tensor
        An integer tensor of shape (R,). Each element is the length of segments
        of input ``lhs_data``. The summation of all elements must be equal to N.

    Returns
    -------
    tensor
        The output dense matrix of shape (N, D2)
    """
    return segment_mm_internal(lhs_data, rhs_data, seglen_lhs)

def gather_mm(lhs_data, rhs_data, idx_lhs = None, idx_rhs = None):
    r"""Gather data according to the given indices and perform matrix multiplication.

    Let the result tensor be C, the operator conducts the following computation:

    If both idx_lhs and idx_rhs are not none:

      c[i] = lhs_data[idx_lhs[i]] @ rhs_data[idx_rhs[i]]
      , where len(C) == len(idx_lhs) == len(idx_rhs)

    If idx_lhs is given but not idx_rhs:

      c[i] = rhs_data[idx_lhs[i]] @ rhs_data[i]
      , where len(C) == len(idx_lhs)

    If idx_rhs is given but not idx_lhs:

      c[i] = lhs_data[i] @ rhs_data[idx_rhs[i]]
      , where len(C) == len(idx_rhs)


    Parameters
    ----------
    lhs_data : tensor
        2-D tensor of shape (N, D1)
    rhs_data : tensor
        3-D tensor of shape (R, D1, D2)
    idx_lhs : Tensor, optional
        If specified, must be a 1-D integer tensor of shape (K,).
    idx_rhs : Tensor, optional
        If specified, must be a 1-D integer tensor of shape (K,).

    Returns
    -------
    Tensor
        The output dense matrix of shape (N, D2)
    """
    return gather_mm_internal(lhs_data, rhs_data, idx_lhs, idx_rhs)
