"""DGL sparse matrix module."""
from typing import Optional, Tuple

import torch

__all__ = [
    "SparseMatrix",
    "create_from_coo",
    "create_from_csr",
    "create_from_csc",
]


class SparseMatrix:
    r"""Class for sparse matrix.

    Parameters
    ----------
    row : tensor
        The row indices of shape nnz.
    col : tensor
        The column indices of shape nnz.
    val : tensor, optional
        The values of shape (nnz, *). If None, it will be a tensor of shape (nnz)
        filled by 1.
    shape : tuple[int, int], optional
        Shape or size of the sparse matrix. If not provided the shape will be
        inferred from the row and column indices.

    Examples
    --------
    Case1: Sparse matrix with row indices, col indices and values (scalar).

    >>> src = torch.tensor([1, 1, 2])
    >>> dst = torch.tensor([2, 4, 3])
    >>> val = torch.tensor([1, 1, 1])
    >>> A = SparseMatrix(src, dst, val)
    >>> print(A)
    SparseMatrix(indices=tensor([[1, 1, 2],
        [2, 4, 3]]),
    values=tensor([1, 1, 1]),
    shape=(3, 5), nnz=3)

    Case2: Sparse matrix with row indices, col indices and values (vector).

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3]])
    >>> A = SparseMatrix(src, dst, val)
    >>> print(A)
    SparseMatrix(indices=tensor([[1, 1, 2],
            [2, 4, 3]]),
    values=tensor([[1, 1],
            [2, 2],
            [3, 3]]),
    shape=(3, 5), nnz=3)
    """

    def __init__(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        val: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, int]] = None,
    ):
        if val is None:
            val = torch.ones(row.shape[0])
        i = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
        if shape is None:
            self.adj = torch.sparse_coo_tensor(i, val).coalesce()
        else:
            if len(val.shape) > 1:
                shape += (val.shape[-1],)
            self.adj = torch.sparse_coo_tensor(i, val, shape).coalesce()

    def __repr__(self):
        return f'SparseMatrix(indices={self.indices("COO")}, \nvalues={self.val}, \
                \nshape={self.shape}, nnz={self.nnz})'

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the sparse matrix.

        Returns
        -------
        tuple[int]
            The shape of the matrix
        """
        return (self.adj.shape[0], self.adj.shape[1])

    @property
    def nnz(self) -> int:
        """The number of nonzero elements of the sparse matrix.

        Returns
        -------
        int
            The number of nonzero elements of the matrix
        """
        return self.adj._nnz()

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the values of the sparse matrix.

        Returns
        -------
        torch.dtype
            Data type of the values of the matrix
        """
        return self.adj.dtype

    @property
    def device(self) -> torch.device:
        """Device of the sparse matrix.

        Returns
        -------
        torch.device
            Device of the matrix
        """
        return self.adj.device

    @property
    def row(self) -> torch.Tensor:
        """Get the row indices of the nonzero elements.

        Returns
        -------
        tensor
            Row indices of the nonzero elements
        """
        return self.adj.indices()[0]

    @property
    def col(self) -> torch.Tensor:
        """Get the column indices of the nonzero elements.

        Returns
        -------
        tensor
            Column indices of the nonzero elements
        """
        return self.adj.indices()[1]

    @property
    def val(self) -> torch.Tensor:
        """Get the values of the nonzero elements.

        Returns
        -------
        tensor
            Values of the nonzero elements
        """
        return self.adj.values()

    @val.setter
    def val(self, x: torch.Tensor) -> torch.Tensor:
        """Set the values of the nonzero elements."""
        assert len(x) == self.nnz
        if len(x.shape) == 1:
            shape = self.shape
        else:
            shape = self.shape + (x.shape[-1],)
        self.adj = torch.sparse_coo_tensor(
            self.adj.indices(), x, shape
        ).coalesce()

    def __call__(self, x: torch.Tensor):
        """Create a new sparse matrix with the same sparsity as self but different values.

        Parameters
        ----------
        x : tensor
            Values of the new sparse matrix

        Returns
        -------
        Class object
            A new sparse matrix object of the SparseMatrix class

        """
        assert len(x) == self.nnz
        return SparseMatrix(self.row, self.col, x, shape=self.shape)

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
            return self.adj.indices()
        else:
            raise NotImplementedError

    def coo(self) -> Tuple[torch.Tensor, ...]:
        """Get the coordinate (COO) representation of the sparse matrix.

        Returns
        -------
        tensor
            A tensor containing indices and value tensors.
        """
        return self

    def csr(self) -> Tuple[torch.Tensor, ...]:
        """Get the CSR (Compressed Sparse Row) representation of the sparse matrix.

        Returns
        -------
        tensor
            A tensor containing compressed row pointers, column indices and value tensors.
        """
        return self

    def csc(self) -> Tuple[torch.Tensor, ...]:
        """Get the CSC (Compressed Sparse Column) representation of the sparse matrix.

        Returns
        -------
        tensor
            A tensor containing compressed column pointers, row indices and value tensors.
        """
        return self

    def dense(self) -> torch.Tensor:
        """Get the dense representation of the sparse matrix.

        Returns
        -------
        tensor
            Dense representation of the sparse matrix.
        """
        return self.adj.to_dense()

    def t(self):
        """Alias of :meth:`transpose()`"""
        return self.transpose()

    @property
    def T(self):  # pylint: disable=C0103
        """Alias of :meth:`transpose()`"""
        return self.transpose()

    def transpose(self):
        """Return the transpose of this sparse matrix.

        Returns
        -------
        SparseMatrix
            The transpose of this sparse matrix.

        Example
        -------

        >>> row = torch.tensor([1, 1, 3])
        >>> col = torch.tensor([2, 1, 3])
        >>> val = torch.tensor([1, 1, 2])
        >>> A = create_from_coo(row, col, val)
        >>> A = A.transpose()
        >>> print(A)
        SparseMatrix(indices=tensor([[1, 2, 3],
                [1, 1, 3]]),
        values=tensor([1, 1, 2]),
        shape=(4, 4), nnz=3)
        """
        return SparseMatrix(self.col, self.row, self.val, self.shape[::-1])


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
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of shape (nnz)
        filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`row` and :attr:`col`, i.e.,
        (row.max() + 1, col.max() + 1). Otherwise, :attr:`shape` should be no smaller
        than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix with row and column indices without values.

    >>> src = torch.tensor([1, 1, 2])
    >>> dst = torch.tensor([2, 4, 3])
    >>> A = create_from_coo(src, dst)
    >>> A
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)
    >>> # Specify shape
    >>> A = create_from_coo(src, dst, shape=(5, 5))
    >>> A
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(5, 5), nnz=3)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3]])
    >>> A = create_from_coo(src, dst, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1, 1],
                                [2, 2],
                                [3, 3]]),
                 shape=(3, 5), nnz=3)
    """
    return SparseMatrix(row=row, col=col, val=val, shape=shape)


def create_from_csr(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    """Create a sparse matrix from CSR indices.

    For row i of the sparse matrix

    - the column indices of the nonzero entries are stored in ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : tensor
        Pointer to the column indices of shape (N + 1), where N is the number of rows.
    indices : tensor
        The column indices of shape (nnz).
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of shape (nnz)
        filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`indptr` and :attr:`indices`, i.e.,
        (len(indptr) - 1, indices.max() + 1). Otherwise, :attr:`shape` should be no smaller
        than this.

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
    adj_csr = torch.sparse_csr_tensor(
        indptr, indices, torch.ones(indices.shape[0])
    )
    adj_coo = adj_csr.to_sparse_coo().coalesce()
    row, col = adj_coo.indices()

    return SparseMatrix(row=row, col=col, val=val, shape=shape)


def create_from_csc(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    """Create a sparse matrix from CSC indices.

    For column i of the sparse matrix

    - the row indices of the nonzero entries are stored in ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : tensor
        Pointer to the row indices of shape N + 1, where N is the number of columns.
    indices : tensor
        The row indices of shape nnz.
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of shape (nnz)
        filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`indptr` and :attr:`indices`, i.e.,
        (indices.max() + 1, len(indptr) - 1). Otherwise, :attr:`shape` should be no smaller
        than this.

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
    adj_csr = torch.sparse_csr_tensor(
        indptr, indices, torch.ones(indices.shape[0])
    )
    adj_coo = adj_csr.to_sparse_coo().coalesce()
    col, row = adj_coo.indices()

    return SparseMatrix(row=row, col=col, val=val, shape=shape)
