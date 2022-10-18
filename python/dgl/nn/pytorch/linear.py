"""Various commonly used linear modules"""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import math

import torch
import torch.nn as nn

from ...ops import gather_mm, segment_mm

__all__ = ["TypedLinear"]


class TypedLinear(nn.Module):
    r"""Linear transformation according to types.

    For each sample of the input batch :math:`x \in X`, apply linear transformation
    :math:`xW_t`, where :math:`t` is the type of :math:`x`.

    The module supports two regularization methods (basis-decomposition and
    block-diagonal-decomposition) proposed by "`Modeling Relational Data
    with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"

    The basis regularization decomposes :math:`W_t` by:

    .. math::

       W_t^{(l)} = \sum_{b=1}^B a_{tb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{tb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_t` into :math:`B`
    block-diagonal matrices. We refer to :math:`B` as the number of bases:

    .. math::

       W_t^{(l)} = \oplus_{b=1}^B Q_{tb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{tb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)\times(d^{l}/B)}`.

    Parameters
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    num_types : int
        Total number of types.
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Typically smaller
        than ``num_types``.
        Default: ``None``.

    Examples
    --------

    No regularization.

    >>> from dgl.nn import TypedLinear
    >>> import torch
    >>>
    >>> x = torch.randn(100, 32)
    >>> x_type = torch.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    torch.Size([100, 64])

    With basis regularization

    >>> x = torch.randn(100, 32)
    >>> x_type = torch.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5, regularizer='basis', num_bases=4)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    torch.Size([100, 64])
    """

    def __init__(
        self, in_size, out_size, num_types, regularizer=None, num_bases=None
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_types = num_types
        if regularizer is None:
            self.W = nn.Parameter(torch.Tensor(num_types, in_size, out_size))
        elif regularizer == "basis":
            if num_bases is None:
                raise ValueError(
                    'Missing "num_bases" for basis regularization.'
                )
            self.W = nn.Parameter(torch.Tensor(num_bases, in_size, out_size))
            self.coeff = nn.Parameter(torch.Tensor(num_types, num_bases))
            self.num_bases = num_bases
        elif regularizer == "bdd":
            if num_bases is None:
                raise ValueError('Missing "num_bases" for bdd regularization.')
            if in_size % num_bases != 0 or out_size % num_bases != 0:
                raise ValueError(
                    "Input and output sizes must be divisible by num_bases."
                )
            self.submat_in = in_size // num_bases
            self.submat_out = out_size // num_bases
            self.W = nn.Parameter(
                torch.Tensor(
                    num_types, num_bases * self.submat_in * self.submat_out
                )
            )
            self.num_bases = num_bases
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}'
            )
        self.regularizer = regularizer
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters"""
        with torch.no_grad():
            # Follow torch.nn.Linear 's initialization to use kaiming_uniform_ on in_size
            if self.regularizer is None:
                nn.init.uniform_(
                    self.W,
                    -1 / math.sqrt(self.in_size),
                    1 / math.sqrt(self.in_size),
                )
            elif self.regularizer == "basis":
                nn.init.uniform_(
                    self.W,
                    -1 / math.sqrt(self.in_size),
                    1 / math.sqrt(self.in_size),
                )
                nn.init.xavier_uniform_(
                    self.coeff, gain=nn.init.calculate_gain("relu")
                )
            elif self.regularizer == "bdd":
                nn.init.uniform_(
                    self.W,
                    -1 / math.sqrt(self.submat_in),
                    1 / math.sqrt(self.submat_in),
                )
            else:
                raise ValueError(
                    f'Supported regularizer options: "basis", "bdd", but got {regularizer}'
                )

    def get_weight(self):
        """Get type-wise weight"""
        if self.regularizer is None:
            return self.W
        elif self.regularizer == "basis":
            W = self.W.view(self.num_bases, self.in_size * self.out_size)
            return (self.coeff @ W).view(
                self.num_types, self.in_size, self.out_size
            )
        elif self.regularizer == "bdd":
            return self.W
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}'
            )

    def forward(self, x, x_type, sorted_by_type=False):
        """Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            A 2D input tensor. Shape: (N, D1)
        x_type : torch.Tensor
            A 1D integer tensor storing the type of the elements in ``x`` with one-to-one
            correspondenc. Shape: (N,)
        sorted_by_type : bool, optional
            Whether the inputs have been sorted by the types. Forward on pre-sorted inputs may
            be faster.

        Returns
        -------
        y : torch.Tensor
            The transformed output tensor. Shape: (N, D2)
        """
        w = self.get_weight()
        if self.regularizer == "bdd":
            w = w.index_select(0, x_type).view(
                -1, self.submat_in, self.submat_out
            )
            x = x.view(-1, 1, self.submat_in)
            return torch.bmm(x, w).view(-1, self.out_size)
        elif sorted_by_type:
            pos_l = torch.searchsorted(
                x_type, torch.arange(self.num_types, device=x.device)
            )
            pos_r = torch.cat(
                [pos_l[1:], torch.tensor([len(x_type)], device=x.device)]
            )
            seglen = (
                pos_r - pos_l
            ).cpu()  # XXX(minjie): cause device synchronize
            return segment_mm(x, w, seglen_a=seglen)
        else:
            return gather_mm(x, w, idx_b=x_type)

    def __repr__(self):
        if self.regularizer is None:
            return (
                f"TypedLinear(in_size={self.in_size}, out_size={self.out_size}, "
                f"num_types={self.num_types})"
            )
        else:
            return (
                f"TypedLinear(in_size={self.in_size}, out_size={self.out_size}, "
                f"num_types={self.num_types}, regularizer={self.regularizer}, "
                f"num_bases={self.num_bases})"
            )
