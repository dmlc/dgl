"""DGL diagonal matrix module."""
from typing import Optional, Tuple

import torch

from .sparse_matrix import SparseMatrix, create_from_coo


class DiagMatrix:
    """Diagonal Matrix Class

    Parameters
    ----------
    val : torch.Tensor
        Diagonal of the matrix. It can take shape (N) or (N, D).
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`val`, i.e.,
        (N, N). Otherwise, :attr:`len(val)` must be equal to :attr:`min(shape)`.

    Attributes
    ----------
    val : torch.Tensor
        Diagonal of the matrix.
    shape : tuple[int, int]
        Shape of the matrix.
    """

    def __init__(
        self, val: torch.Tensor, shape: Optional[Tuple[int, int]] = None
    ):
        len_val = len(val)
        if shape is not None:
            assert len_val == min(shape), (
                f"Expect len(val) to be min(shape), got {len_val} for len(val)"
                "and {shape} for shape."
            )
        else:
            shape = (len_val, len_val)
        self.val = val
        self.shape = shape

    def __repr__(self):
        return f"DiagMatrix(val={self.val}, \nshape={self.shape})"

    @property
    def nnz(self) -> int:
        """Return the number of non-zero values in the matrix

        Returns
        -------
        int
            The number of non-zero values in the matrix
        """
        return self.val.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the matrix

        Returns
        -------
        torch.dtype
            Data type of the matrix
        """
        return self.val.dtype

    @property
    def device(self) -> torch.device:
        """Return the device of the matrix

        Returns
        -------
        torch.device
            Device of the matrix
        """
        return self.val.device

    def as_sparse(self) -> SparseMatrix:
        """Convert the diagonal matrix into a sparse matrix object

        Returns
        -------
        SparseMatrix
            The converted sparse matrix object

        Example
        -------

        >>> import torch
        >>> val = torch.ones(5)
        >>> mat = diag(val)
        >>> sp_mat = mat.as_sparse()
        >>> print(sp_mat)
        SparseMatrix(indices=tensor([[0, 1, 2, 3, 4],
                                     [0, 1, 2, 3, 4]]),
                     values=tensor([1., 1., 1., 1., 1.]),
                     shape=(5, 5), nnz=5)
        """
        row = col = torch.arange(len(self.val)).to(self.device)
        return create_from_coo(row=row, col=col, val=self.val, shape=self.shape)

    def t(self):
        """Alias of :meth:`transpose()`"""
        return self.transpose()

    @property
    def T(self):  # pylint: disable=C0103
        """Alias of :meth:`transpose()`"""
        return self.transpose()

    def transpose(self):
        """Return the transpose of the matrix.

        Returns
        -------
        DiagMatrix
            The transpose of the matrix.

        Example
        --------

        >>> val = torch.arange(1, 5).float()
        >>> mat = diag(val, shape=(4, 5))
        >>> mat = mat.transpose()
        >>> print(mat)
        DiagMatrix(val=tensor([1., 2., 3., 4.]),
        shape=(5, 4))
        """
        return DiagMatrix(self.val, self.shape[::-1])


def diag(
    val: torch.Tensor, shape: Optional[Tuple[int, int]] = None
) -> DiagMatrix:
    """Create a diagonal matrix based on the diagonal values

    Parameters
    ----------
    val : torch.Tensor
        Diagonal of the matrix. It can take shape (N) or (N, D).
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`val`, i.e.,
        (N, N). Otherwise, :attr:`len(val)` must be equal to :attr:`min(shape)`.

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------

    Case1: 5-by-5 diagonal matrix with scaler values on the diagonal

    >>> import torch
    >>> val = torch.ones(5)
    >>> mat = diag(val)
    >>> print(mat)
    DiagMatrix(val=tensor([1., 1., 1., 1., 1.]),
               shape=(5, 5))

    Case2: 5-by-10 diagonal matrix with scaler values on the diagonal

    >>> val = torch.ones(5)
    >>> mat = diag(val, shape=(5, 10))
    >>> print(mat)
    DiagMatrix(val=tensor([1., 1., 1., 1., 1.]),
               shape=(5, 10))

    Case3: 5-by-5 diagonal matrix with tensor values on the diagonal

    >>> val = torch.randn(5, 3)
    >>> mat = diag(val)
    >>> mat.shape
    (5, 5)
    >>> mat.nnz
    5
    """
    # NOTE(Mufei): this may not be needed if DiagMatrix is simple enough
    return DiagMatrix(val, shape)


def identity(
    shape: Tuple[int, int],
    d: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> DiagMatrix:
    """Create a diagonal matrix with ones on the diagonal and zeros elsewhere

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the matrix.
    d : int, optional
        If None, the diagonal entries will be scaler 1. Otherwise, the diagonal
        entries will be a 1-valued tensor of shape (d).
    dtype : torch.dtype, optional
        The data type of the matrix
    device : torch.device, optional
        The device of the matrix

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------

    Case1: 3-by-3 matrix with scaler diagonal values

    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

    >>> mat = identity(shape=(3, 3))
    >>> print(mat)
    DiagMatrix(val=tensor([1., 1., 1.]),
               shape=(3, 3))

    Case2: 3-by-5 matrix with scaler diagonal values

    [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0]]

    >>> mat = identity(shape=(3, 5))
    >>> print(mat)
    DiagMatrix(val=tensor([1., 1., 1.]),
               shape=(3, 5))

    Case3: 3-by-3 matrix with tensor diagonal values

    >>> mat = identity(shape=(3, 3), d=2)
    >>> print(mat)
    DiagMatrix(val=tensor([[1., 1.],
            [1., 1.],
            [1., 1.]]),
    shape=(3, 3))
    """
    len_val = min(shape)
    if d is None:
        val_shape = (len_val,)
    else:
        val_shape = (len_val, d)
    val = torch.ones(val_shape, dtype=dtype, device=device)
    return diag(val, shape)
