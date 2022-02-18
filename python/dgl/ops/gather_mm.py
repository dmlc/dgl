"""dgl gather_mm operator module."""
from .. import backend as F

__all__ = ['gather_mm']

def gather_mm(a, b, idx_a=None, idx_b=None):
    r"""Gather data according to the given indices and perform matrix multiplication.

    Let the result tensor be ``c``, the operator conducts the following computation:

    If both ``idx_a`` and ``idx_b`` are not none::

      c[i] = a[idx_a[i]] @ b[idx_b[i]]
      , where len(c) == len(idx_a) == len(idx_b)

    If ``idx_a`` is given but not ``idx_b``::

      c[i] = b[idx_a[i]] @ b[i]
      , where len(c) == len(idx_a)

    If ``idx_b is given but not idx_a``::

      c[i] = a[i] @ b[idx_b[i]]
      , where len(c) == len(idx_b)


    Parameters
    ----------
    a : Tensor
        2-D tensor of shape ``(N, D1)``
    b : Tensor
        3-D tensor of shape ``(R, D1, D2)``
    idx_a : Tensor, optional
        If specified, must be a 1-D integer tensor of shape ``(K,)``.
    idx_b : Tensor, optional
        If specified, must be a 1-D integer tensor of shape ``(K,)``.

    Returns
    -------
    Tensor
        The output dense matrix of shape ``(N, D2)``
    """
    if idx_a is None and idx_b is None:
        raise ValueError('gather_mm operator requires at least one of idx_a or idx_b given.')
    N, D1 = F.shape(a)
    R, _, D2 = F.shape(b)
    if ((idx_a is None and idx_b is not None) and
        (N > 1000000 or D1 > 8 or D2 > 8)):
        # Use segment_mm for large workload
        import torch
        sorted_idx_b, perm = torch.sort(idx_b)
        sorted_a = a[perm]
        pos_l = torch.searchsorted(sorted_idx_b, torch.arange(R, device=a.device))
        pos_r = torch.cat([pos_l[1:], torch.tensor([len(idx_b)], device=a.device)])
        seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize
        return F.segment_mm(sorted_a, b, seglen)
    else:
        return F.gather_mm(a, b, idx_a, idx_b)
