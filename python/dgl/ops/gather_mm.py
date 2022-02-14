"""dgl edge_softmax operator module."""
from ..backend import gather_mm as gather_mm_internal
from ..backend import segment_mm as segment_mm_internal

__all__ = ['gather_mm', 'segment_mm']

def segment_mm(a, b, seglen_a):
    r""" Performs matrix multiplication according to segments.
    Suppose ``seglen_a == [10, 5, 0, 3]``, the operator will perform
    four matrix multiplications:
    a[0:10] @ b[0], a[10:15] @ b[1], a[15:15] @ b[2], a[15:18] @ b[3]

    Parameters
    ----------
    a : tensor
        2-D tensor of shape (N, D1)
    b : tensor
        2-D tensor of shape (R * D1, D2)
    seglen_a : Tensor
        An integer tensor of shape (R,). Each element is the length of segments
        of input ``a``. The summation of all elements must be equal to N.

    Returns
    -------
    Tensor
        The output dense matrix of shape (N, D2)
    """
    return segment_mm_internal(a, b, seglen_a)

def gather_mm(a, b, idx_a = None, idx_b = None):
    r"""Gather data according to the given indices and perform matrix multiplication.

    Let the result tensor be C, the operator conducts the following computation:

    If both idx_a and idx_b are not none:

      c[i] = a[idx_a[i]] @ b[idx_b[i]]
      , where len(C) == len(idx_a) == len(idx_b)

    If idx_a is given but not idx_b:

      c[i] = b[idx_a[i]] @ b[i]
      , where len(C) == len(idx_a)

    If idx_b is given but not idx_a:

      c[i] = a[i] @ a[idx_b[i]]
      , where len(C) == len(idx_b)


    Parameters
    ----------
    a : tensor
        2-D tensor of shape (N, D1)
    b : tensor
        3-D tensor of shape (R, D1, D2)
    idx_a : Tensor, optional
        If specified, must be a 1-D integer tensor of shape (K,).
    idx_b : Tensor, optional
        If specified, must be a 1-D integer tensor of shape (K,).

    Returns
    -------
    Tensor
        The output dense matrix of shape (N, D2)
    """
    return gather_mm_internal(a, b, idx_a, idx_b)
