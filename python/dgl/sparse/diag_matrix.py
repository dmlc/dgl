"""DGL diagonal matrix module."""
# pylint: disable= invalid-name
from typing import Optional, Tuple

import torch

from .sparse_matrix import from_coo, SparseMatrix


class DiagMatrix:
    r"""Class for diagonal matrix.

    Parameters
    ----------
    val : torch.Tensor
        Diagonal of the matrix, in shape ``(N)`` or ``(N, D)``
    shape : tuple[int, int], optional
        If specified, :attr:`len(val)` must be equal to :attr:`min(shape)`,
        otherwise, it will be inferred from :attr:`val`, i.e., ``(N, N)``
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

    def __repr__(self):
        return _diag_matrix_str(self)

    @property
    def val(self) -> torch.Tensor:
        """Returns the values of the non-zero elements.

        Returns
        -------
        torch.Tensor
            Values of the non-zero elements
        """
        return self._val

    @property
    def shape(self) -> Tuple[int]:
        """Returns the shape of the diagonal matrix.

        Returns
        -------
        Tuple[int]
            The shape of the diagonal matrix
        """
        return self._shape

    @property
    def nnz(self) -> int:
        """Returns the number of non-zero elements in the diagonal matrix.

        Returns
        -------
        int
            The number of non-zero elements in the diagonal matrix
        """
        return self.val.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the diagonal matrix.

        Returns
        -------
        torch.dtype
            Data type of the diagonal matrix
        """
        return self.val.dtype

    @property
    def device(self) -> torch.device:
        """Returns the device the diagonal matrix is on.

        Returns
        -------
        torch.device
            The device the diagonal matrix is on
        """
        return self.val.device

    def to_sparse(self) -> SparseMatrix:
        """Returns a copy in sparse matrix format of the diagonal matrix.

        Returns
        -------
        SparseMatrix
            The copy in sparse matrix format

        Examples
        --------

        >>> import torch
        >>> val = torch.ones(5)
        >>> D = dglsp.diag(val)
        >>> D.to_sparse()
        SparseMatrix(indices=tensor([[0, 1, 2, 3, 4],
                                     [0, 1, 2, 3, 4]]),
                     values=tensor([1., 1., 1., 1., 1.]),
                     shape=(5, 5), nnz=5)
        """
        row = col = torch.arange(len(self.val)).to(self.device)
        return from_coo(row=row, col=col, val=self.val, shape=self.shape)

    def to_dense(self) -> torch.Tensor:
        """Returns a copy in dense matrix format of the diagonal matrix.

        Returns
        -------
        torch.Tensor
            The copy in dense matrix format
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
        """Returns a matrix that is a transposed version of the diagonal matrix.

        Returns
        -------
        DiagMatrix
            The transpose of the matrix

        Examples
        --------

        >>> val = torch.arange(1, 5).float()
        >>> D = dglsp.diag(val, shape=(4, 5))
        >>> D.transpose()
        DiagMatrix(val=tensor([1., 2., 3., 4.]),
                   shape=(5, 4))
        """
        return DiagMatrix(self.val, self.shape[::-1])

    def to(self, device=None, dtype=None):
        """Performs matrix dtype and/or device conversion. If the target device
        and dtype are already in use, the original matrix will be returned.

        Parameters
        ----------
        device : torch.device, optional
            The target device of the matrix if provided, otherwise the current
            device will be used
        dtype : torch.dtype, optional
            The target data type of the matrix values if provided, otherwise the
            current data type will be used

        Returns
        -------
        DiagMatrix
            The converted matrix

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.to(device="cuda:0", dtype=torch.int32)
        DiagMatrix(values=tensor([1, 1], device='cuda:0', dtype=torch.int32),
                   shape=(2, 2))
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        if device == self.device and dtype == self.dtype:
            return self

        return diag(self.val.to(device=device, dtype=dtype), self.shape)

    def cuda(self):
        """Moves the matrix to GPU. If the matrix is already on GPU, the
        original matrix will be returned. If multiple GPU devices exist,
        ``cuda:0`` will be selected.

        Returns
        -------
        DiagMatrix
            The matrix on GPU

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.cuda()
        DiagMatrix(values=tensor([1., 1.], device='cuda:0'),
                   shape=(2, 2))
        """
        return self.to(device="cuda")

    def cpu(self):
        """Moves the matrix to CPU. If the matrix is already on CPU, the
        original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix on CPU

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.cpu()
        DiagMatrix(values=tensor([1., 1.]),
                   shape=(2, 2))
        """
        return self.to(device="cpu")

    def float(self):
        """Converts the matrix values to float32 data type. If the matrix
        already uses float data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with float values

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.float()
        DiagMatrix(values=tensor([1., 1.]),
                   shape=(2, 2))
        """
        return self.to(dtype=torch.float)

    def double(self):
        """Converts the matrix values to double data type. If the matrix already
        uses double data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with double values

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.double()
        DiagMatrix(values=tensor([1., 1.], dtype=torch.float64),
                   shape=(2, 2))
        """
        return self.to(dtype=torch.double)

    def int(self):
        """Converts the matrix values to int32 data type. If the matrix already
        uses int data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with int values

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.int()
        DiagMatrix(values=tensor([1, 1], dtype=torch.int32),
                   shape=(2, 2))
        """
        return self.to(dtype=torch.int)

    def long(self):
        """Converts the matrix values to long data type. If the matrix already
        uses long data type, the original matrix will be returned.

        Returns
        -------
        DiagMatrix
            The matrix with long values

        Examples
        --------

        >>> val = torch.ones(2)
        >>> D = dglsp.diag(val)
        >>> D.long()
        DiagMatrix(values=tensor([1, 1]),
                   shape=(2, 2))
        """
        return self.to(dtype=torch.long)


def diag(
    val: torch.Tensor, shape: Optional[Tuple[int, int]] = None
) -> DiagMatrix:
    """Creates a diagonal matrix based on the diagonal values.

    Parameters
    ----------
    val : torch.Tensor
        Diagonal of the matrix, in shape ``(N)`` or ``(N, D)``
    shape : tuple[int, int], optional
        If specified, :attr:`len(val)` must be equal to :attr:`min(shape)`,
        otherwise, it will be inferred from :attr:`val`, i.e., ``(N, N)``

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------

    Case1: 5-by-5 diagonal matrix with scaler values on the diagonal

    >>> import torch
    >>> val = torch.ones(5)
    >>> dglsp.diag(val)
    DiagMatrix(val=tensor([1., 1., 1., 1., 1.]),
               shape=(5, 5))

    Case2: 5-by-10 diagonal matrix with scaler values on the diagonal

    >>> val = torch.ones(5)
    >>> dglsp.diag(val, shape=(5, 10))
    DiagMatrix(val=tensor([1., 1., 1., 1., 1.]),
               shape=(5, 10))

    Case3: 5-by-5 diagonal matrix with vector values on the diagonal

    >>> val = torch.randn(5, 3)
    >>> D = dglsp.diag(val)
    >>> D.shape
    (5, 5)
    >>> D.nnz
    5
    """
    assert (
        val.dim() <= 2
    ), "The values of a DiagMatrix can only be scalars or vectors."
    # NOTE(Mufei): this may not be needed if DiagMatrix is simple enough
    return DiagMatrix(val, shape)


def identity(
    shape: Tuple[int, int],
    d: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> DiagMatrix:
    r"""Creates a diagonal matrix with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the matrix.
    d : int, optional
        If None, the diagonal entries will be scaler 1. Otherwise, the diagonal
        entries will be a 1-valued tensor of shape ``(d)``.
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

    .. code::

        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]

    >>> dglsp.identity(shape=(3, 3))
    DiagMatrix(val=tensor([1., 1., 1.]),
               shape=(3, 3))

    Case2: 3-by-5 matrix with scaler diagonal values

    .. code::

        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0]]

    >>> dglsp.identity(shape=(3, 5))
    DiagMatrix(val=tensor([1., 1., 1.]),
               shape=(3, 5))

    Case3: 3-by-3 matrix with vector diagonal values

    >>> dglsp.identity(shape=(3, 3), d=2)
    DiagMatrix(values=tensor([[1., 1.],
                              [1., 1.],
                              [1., 1.]]),
               shape=(3, 3), val_size=(2,))
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
    representation.
    """
    values_str = str(spmat.val)
    meta_str = f"shape={spmat.shape}"
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
