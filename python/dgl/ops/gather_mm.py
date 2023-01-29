"""dgl gather_mm operator module."""
from .. import backend as F

__all__ = ["gather_mm"]


def gather_mm(a, b, *, idx_b):
    r"""Gather data according to the given indices and perform matrix multiplication.

    Let the result tensor be ``c``, the operator conducts the following computation:

      c[i] = a[i] @ b[idx_b[i]]
      , where len(c) == len(idx_b)


    Parameters
    ----------
    a : Tensor
        A 2-D tensor of shape ``(N, D1)``
    b : Tensor
        A 3-D tensor of shape ``(R, D1, D2)``
    idx_b : Tensor, optional
        An 1-D integer tensor of shape ``(N,)``.

    Returns
    -------
    Tensor
        The output dense matrix of shape ``(N, D2)``
    """
    N, D1 = F.shape(a)
    R, _, D2 = F.shape(b)
    if N > 1000000 or D1 > 8 or D2 > 8:
        # Use segment_mm for large workload
        import torch

        sorted_idx_b, perm = torch.sort(idx_b)
        _, rev_perm = torch.sort(perm)
        sorted_a = torch.index_select(a, 0, perm)
        pos_l = torch.searchsorted(
            sorted_idx_b, torch.arange(R, device=a.device)
        )
        pos_r = torch.cat(
            [pos_l[1:], torch.tensor([len(idx_b)], device=a.device)]
        )
        seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize
        return torch.index_select(
            F.segment_mm(sorted_a, b, seglen), 0, rev_perm
        )
    else:
        return F.gather_mm(a, b, None, idx_b)
