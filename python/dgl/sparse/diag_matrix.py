"""DGL diagonal matrix module."""
# pylint: disable= invalid-name
from typing import Optional, Tuple

import torch

from .sparse_matrix import from_coo, SparseMatrix


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
        self._val = val
        self._shape = shape

    @property
    def val(self) -> torch.Tensor:
        """Get the values of the nonzero elements.

        Returns
        -------
        torch.Tensor
            Values of the nonzero elements
        """
        return self._val

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the sparse matrix.

        Returns
        -------
        Tuple[int]
            The shape of the matrix
        """
        return self._shape

    def __repr__(self):
        return _diag_matrix_str(self)

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

    def to_sparse(self) -> SparseMatrix:
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
        >>> sp_mat = mat.to_sparse()
        >>> print(sp_mat)
        SparseMatrix(indices=tensor([[0, 1, 2, 3, 4],
                                     [0, 1, 2, 3, 4]]),
                     values=tensor([1., 1., 1., 1., 1.]),
                     shape=(5, 5), nnz=5)
        """
        row = col = torch.arange(len(self.val)).to(self.device)
        return from_coo(row=row, col=col, val=self.val, shape=self.shape)

    def to_dense(self) -> torch.Tensor:
        """Return a dense representation of the matrix.

        Returns
        -------
        torch.Tensor
            Dense representation of the diagonal matrix.
        """
        val = self.val
        device = self.device
        shape = self.shape + val.shape[1:]
        mat = torch.zeros(shape, device=device, dtype=self.dtype)
        row = col = torch.arange(len(val)).to(device)
        mat[row, col] = val
        return mat

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

    def to(self, device=None, dtype=None):
        """Perform matrix dtype and/or device conversion. If the target device
        and dtype are already in use, the original matrix will be returned.

        Parameters
        ----------
        device : torch.device, optional
            The target device of the matrix if given, otherwise the current
            device will be used
        dtype : torch.dtype, optional
            The target data type of the matrix values if given, otherwise the
            current data type will be used

        Returns
        -------
        DiagMatrix
            The result matrix

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.to(device='cuda:0', dtype=torch.int32)
        DiagMatrix(values=tensor([1, 1], device='cuda:0', dtype=torch.int32),
                   size=(2, 2))
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        if device == self.device and dtype == self.dtype:
            return self

        return diag(self.val.to(device=device, dtype=dtype), self.shape)

    def cuda(self):
        """Move the matrix to GPU. If the matrix is already on GPU, the
        original matrix will be returned. If multiple GPU devices exist,
        'cuda:0' will be selected.

        Returns
        -------
        DiagMatrix
            The matrix on GPU

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.cuda()
        DiagMatrix(values=tensor([1., 1.], device='cuda:0'),
                   size=(2, 2))
        """
        return self.to(device="cuda")

    def cpu(self):
        """Move the matrix to CPU. If the matrix is already on CPU, the
        original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix on CPU

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.cpu()
        DiagMatrix(values=tensor([1., 1.]),
                   size=(2, 2))
        """
        return self.to(device="cpu")

    def float(self):
        """Convert the matrix values to float data type. If the matrix already
        uses float data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with float values

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.float()
        DiagMatrix(values=tensor([1., 1.]),
                   size=(2, 2))
        """
        return self.to(dtype=torch.float)

    def double(self):
        """Convert the matrix values to double data type. If the matrix already
        uses double data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with double values

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.double()
        DiagMatrix(values=tensor([1., 1.], dtype=torch.float64),
                   size=(2, 2))
        """
        return self.to(dtype=torch.double)

    def int(self):
        """Convert the matrix values to int data type. If the matrix already
        uses int data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with int values

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.int()
        DiagMatrix(values=tensor([1, 1], dtype=torch.int32),
                   size=(2, 2))
        """
        return self.to(dtype=torch.int)

    def long(self):
        """Convert the matrix values to long data type. If the matrix already
        uses long data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with long values

        Example
        --------

        >>> val = torch.ones(2)
        >>> mat = diag(val)
        >>> mat.long()
        DiagMatrix(values=tensor([1, 1]),
                   size=(2, 2))
        """
        return self.to(dtype=torch.long)


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


def _diag_matrix_str(spmat: DiagMatrix) -> str:
    """Internal function for converting a diagonal matrix to string
    representation."""
    values_str = str(spmat.val)
    meta_str = f"size={spmat.shape}"
    if spmat.val.dim() > 1:
        val_size = tuple(spmat.val.shape[1:])
        meta_str += f", val_size={val_size}"
    prefix = f"{type(spmat).__name__}("

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    final_str = (
        "values="
        + _add_indent(values_str, len("values="))
        + ",\n"
        + meta_str
        + ")"
    )
    final_str = prefix + _add_indent(final_str, len(prefix))
    return final_str
