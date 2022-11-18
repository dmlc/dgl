"""DGL sparse matrix module."""
from typing import Optional, Tuple

import torch


class SparseMatrix:
    r"""Class for sparse matrix."""

    def __init__(self, c_sparse_matrix: torch.ScriptObject):
        self.c_sparse_matrix = c_sparse_matrix

    @property
    def val(self) -> torch.Tensor:
        """Get the values of the nonzero elements.

        Returns
        -------
        torch.Tensor
            Values of the nonzero elements
        """
        return self.c_sparse_matrix.val()

    @val.setter
    def val(self, x: torch.Tensor):
        """Set the non-zero values inplace.

        Parameters
        ----------
        x : torch.Tensor, optional
            The values of shape (nnz) or (nnz, D)
        """
        self.c_sparse_matrix.set_val(x)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the sparse matrix.

        Returns
        -------
        Tuple[int]
            The shape of the matrix
        """
        return tuple(self.c_sparse_matrix.shape())

    @property
    def nnz(self) -> int:
        """The number of nonzero elements of the sparse matrix.

        Returns
        -------
        int
            The number of nonzero elements of the matrix
        """
        return self.c_sparse_matrix.nnz()

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the values of the sparse matrix.

        Returns
        -------
        torch.dtype
            Data type of the values of the matrix
        """
        # FIXME: find a proper way to pass dtype from C++ to Python
        return self.c_sparse_matrix.val().dtype

    @property
    def device(self) -> torch.device:
        """Device of the sparse matrix.

        Returns
        -------
        torch.device
            Device of the matrix
        """
        return self.c_sparse_matrix.device()

    def indices(
        self, fmt: str, return_shuffle=False
    ) -> Tuple[torch.Tensor, ...]:
        """Get the indices of the nonzero elements.

        Parameters
        ----------
        fmt : str
            Sparse matrix storage format. Can be COO or CSR or CSC.
        return_shuffle: bool
            If true, return an extra array of the nonzero value IDs

        Returns
        -------
        tensor
            Indices of the nonzero elements
        """
        if fmt == "COO" and not return_shuffle:
            row, col, _ = self.coo()
            return torch.stack([row, col])
        else:
            raise NotImplementedError

    def __repr__(self):
        return f'SparseMatrix(indices={self.indices("COO")}, \
                \nvalues={self.val}, \nshape={self.shape}, nnz={self.nnz})'

    def coo(self) -> Tuple[torch.Tensor, ...]:
        """Get the coordinate (COO) representation of the sparse matrix.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of tensors containing row, column coordinates and values.
        """
        return self.c_sparse_matrix.coo()

    def csr(self) -> Tuple[torch.Tensor, ...]:
        """Get the coordinate (COO) representation of the sparse matrix.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of tensors containing row, column coordinates and values.
        """
        return self.c_sparse_matrix.csr()

    def csc(self) -> Tuple[torch.Tensor, ...]:
        """Get the coordinate (COO) representation of the sparse matrix.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of tensors containing row, column coordinates and values.
        """
        return self.c_sparse_matrix.csc()

    def dense(self) -> torch.Tensor:
        """Return a dense representation of the matrix.

        Returns
        -------
        torch.Tensor
            Dense representation of the sparse matrix.
        """
        row, col, val = self.coo()
        shape = self.shape + val.shape[1:]
        mat = torch.zeros(shape, device=self.device)
        mat[row, col] = val
        return mat


def create_from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    """Create a sparse matrix from row and column coordinates.

    Parameters
    ----------
    row : tensor
        The row indices of shape (nnz).
    col : tensor
        The column indices of shape (nnz).
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of
        shape (nnz) filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`row` and :attr:`col`,
        i.e., (row.max() + 1, col.max() + 1). Otherwise, :attr:`shape` should
        be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix with row and column indices without values.

    >>> dst = torch.tensor([1, 1, 2])
    >>> src = torch.tensor([2, 4, 3])
    >>> A = create_from_coo(dst, src)
    >>> print(A)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)
    >>> # Specify shape
    >>> A = create_from_coo(dst, src, shape=(5, 5))
    >>> print(A)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(5, 5), nnz=3)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
    >>> A = create_from_coo(dst, src, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1, 1],
                                [2, 2],
                                [3, 3]]),
                 shape=(3, 5), nnz=3)
    """
    if shape is None:
        shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)
    if val is None:
        val = torch.ones(row.shape[0]).to(row.device)

    return SparseMatrix(
        torch.ops.dgl_sparse.create_from_coo(row, col, val, shape)
    )


# FIXME: The docstring cannot print A because we cannot print
# the indices of CSR/CSC
def create_from_csr(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    """Create a sparse matrix from CSR indices.

    For row i of the sparse matrix

    - the column indices of the nonzero entries are stored in
      ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : tensor
        Pointer to the column indices of shape (N + 1), where N is the number
        of rows.
    indices : tensor
        The column indices of shape (nnz).
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of
        shape (nnz) filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`indptr` and
        :attr:`indices`, i.e., (len(indptr) - 1, indices.max() + 1). Otherwise,
        :attr:`shape` should be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix without values

    [[0, 1, 0],
     [0, 0, 1],
     [1, 1, 1]]

    >>> indptr = torch.tensor([0, 1, 2, 5])
    >>> indices = torch.tensor([1, 2, 0, 1, 2])
    >>> A = create_from_csr(indptr, indices)
    >>> print(A)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(3, 3), nnz=5)
    >>> # Specify shape
    >>> A = create_from_csr(indptr, indices, shape=(5, 3))
    >>> print(A)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
            [1, 2, 0, 1, 2]]),
    values=tensor([1., 1., 1., 1., 1.]),
    shape=(5, 3), nnz=5)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> A = create_from_csr(indptr, indices, val)
    >>> print(A)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
            [1, 2, 0, 1, 2]]),
    values=tensor([[1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]]),
    shape=(3, 3), nnz=5)
    """
    if shape is None:
        shape = (indptr.shape[0] - 1, torch.max(indices) + 1)
    if val is None:
        val = torch.ones(indices.shape[0]).to(indptr.device)

    return SparseMatrix(
        torch.ops.dgl_sparse.create_from_csr(indptr, indices, val, shape)
    )


# FIXME: The docstring cannot print A because we cannot print
# the indices of CSR/CSC
def create_from_csc(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    """Create a sparse matrix from CSC indices.

    For column i of the sparse matrix

    - the row indices of the nonzero entries are stored in
      ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : tensor
        Pointer to the row indices of shape N + 1, where N is the
        number of columns.
    indices : tensor
        The row indices of shape nnz.
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of
        shape (nnz) filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`indptr` and
        :attr:`indices`, i.e., (indices.max() + 1, len(indptr) - 1). Otherwise,
        :attr:`shape` should be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix without values

    [[0, 1, 0],
     [0, 0, 1],
     [1, 1, 1]]

    >>> indptr = torch.tensor([0, 1, 3, 5])
    >>> indices = torch.tensor([2, 0, 2, 1, 2])
    >>> A = create_from_csc(indptr, indices)
    >>> print(A)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(3, 3), nnz=5)
    >>> # Specify shape
    >>> A = create_from_csc(indptr, indices, shape=(5, 3))
    >>> print(A)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
            [1, 2, 0, 1, 2]]),
    values=tensor([1., 1., 1., 1., 1.]),
    shape=(5, 3), nnz=5)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> A = create_from_csc(indptr, indices, val)
    >>> print(A)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
            [1, 2, 0, 1, 2]]),
    values=tensor([[2, 2],
            [4, 4],
            [1, 1],
            [3, 3],
            [5, 5]]),
    shape=(3, 3), nnz=5)
    """
    if shape is None:
        shape = (torch.max(indices) + 1, indptr.shape[0] - 1)
    if val is None:
        val = torch.ones(indices.shape[0]).to(indptr.device)

    return SparseMatrix(
        torch.ops.dgl_sparse.create_from_csc(indptr, indices, val, shape)
    )
