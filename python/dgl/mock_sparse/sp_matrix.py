"""dgl sparse matrix module."""
from typing import Optional, Tuple
import torch

__all__ = ['SparseMatrix', 'create_from_coo', 'create_from_csr', 'create_from_csc']

class SparseMatrix:
    r'''Class for sparse matrix.

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
    >>> A.shape
    (3,5)
    >>> A.row
    tensor([1, 1, 2])
    >>> A.val
    tensor([1., 1., 1.])
    >>> A.nnz
    3

    Case2: Sparse matrix with row indices, col indices and values (vector).

    >>> ...
    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3]])
    >>> A = SparseMatrix(src, dst, val)
    >>> A.val
    tensor([[1, 1],
            [2, 2],
            [3, 3]])
    '''
    def __init__(self,
        row: torch.Tensor,
        col: torch.Tensor,
        val: Optional[torch.Tensor] = None,
        shape : Optional[Tuple[int, int]] = None
    ):
        self._row = row
        self._col = col

        if val is None:
            val = torch.ones(row.shape[0])
        self._val = val
        i = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), 0)
        if shape is not None:
            self.adj = torch.sparse_coo_tensor(i, val, shape).coalesce()
        else:
            self.adj = torch.sparse_coo_tensor(i, val).coalesce()

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
    def row(self) -> torch.tensor:
        """Get the row indices of the nonzero elements.

        Returns
        -------
        tensor
            Row indices of the nonzero elements
        """
        return self.adj.indices()[0]

    @property
    def col(self) -> torch.tensor:
        """Get the column indices of the nonzero elements.

        Returns
        -------
        tensor
            Column indices of the nonzero elements
        """
        return self.adj.indices()[1]

    @property
    def val(self) -> torch.tensor:
        """Get the values of the nonzero elements.

        Returns
        -------
        tensor
            Values of the nonzero elements
        """
        return self.adj.values()

    @val.setter
    def val(self, x) -> torch.tensor:
        """Set the values of the nonzero elements."""
        assert len(x) == self.nnz
        self._val = x
        if len(x.shape) == 1:
            shape = self.shape
        else:
            shape = self.shape + (x.shape[-1],)
        self.adj = torch.sparse_coo_tensor(self.adj.indices(), x, shape).coalesce()

    def __call__(self, x):
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

    def indices(self, fmt, return_shuffle=False) -> Tuple[torch.tensor, ...]:
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
        if fmt == 'COO' and not return_shuffle:
            return self.adj.indices()
        else:
            raise NotImplementedError

    def coo(self) -> Tuple[torch.tensor, ...]:
        """Get the coordinate (COO) representation of the sparse matrix.

        Returns
        -------
        tensor
            A tensor containing indices and value tensors.
        """
        return self

    def csr(self) -> Tuple[torch.tensor, ...]:
        """Get the CSR (Compressed Sparse Row) representation of the sparse matrix.

        Returns
        -------
        tensor
            A tensor containing compressed row pointers, column indices and value tensors.
        """
        return self

    def csc(self) -> Tuple[torch.tensor, ...]:
        """Get the CSC (Compressed Sparse Column) representation of the sparse matrix.

        Returns
        -------
        tensor
            A tensor containing compressed column pointers, row indices and value tensors.
        """
        return self

    def dense(self) -> torch.tensor:
        """Get the dense representation of the sparse matrix.

        Returns
        -------
        tensor
            Dense representation of the sparse matrix.
        """
        return self.adj.to_dense()

def create_from_coo(row: torch.Tensor,
                    col: torch.Tensor,
                    val: Optional[torch.Tensor] = None) -> SparseMatrix:
    """Create a sparse matrix from row and column coordinates.

    Parameters
    ----------
    row : tensor
        The row indices of shape nnz.
    col : tensor
        The column indices of shape nnz.
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of shape (nnz)
        filled by 1.

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
    return SparseMatrix(row, col, val)

def create_from_csr(indptr: torch.Tensor,
                    indices: torch.Tensor,
                    val: Optional[torch.Tensor] = None) -> SparseMatrix:
    """Create a sparse matrix from CSR indices.

    For row i of the sparse matrix

    - the column indices of the nonzero entries are stored in ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : tensor
        Pointer to the column indices of shape N + 1, where N is the number of rows.
    indices : tensor
        The column indices of shape nnz.
    val : tensor, optional
        The values of shape (nnz) or (nnz, D). If None, it will be a tensor of shape (nnz)
        filled by 1.

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
    >>> A.shape
    (3, 3)
    >>> A.row
    tensor([0, 1, 2, 2, 2])
    >>> A.val
    tensor([1., 1., 1., 1., 1.])
    >>> A.nnz
    5

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> A = create_from_csr(indptr, indices, val)
    >>> A.val
    tensor([[1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]])
    """
    adj_csr = torch.sparse_csr_tensor(indptr, indices, torch.ones(indices.shape[0]))
    adj_coo = adj_csr.to_sparse_coo().coalesce()
    row, col = adj_coo.indices()

    return SparseMatrix(row, col, val)

def create_from_csc(indptr: torch.Tensor,
                    indices: torch.Tensor,
                    val: Optional[torch.Tensor] = None) -> SparseMatrix:
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
    >>> A.shape
    (3, 3)
    >>> A.row
    tensor([0, 1, 2, 2, 2])
    >>> A.val
    tensor([1., 1., 1., 1., 1.])
    >>> A.nnz
    5

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> A = create_from_csc(indptr, indices, val)
    >>> A.val
    tensor([[1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]])
    """
    adj_csr = torch.sparse_csr_tensor(indptr, indices, torch.ones(indices.shape[0]))
    adj_coo = adj_csr.to_sparse_coo().coalesce()
    col, row = adj_coo.indices()

    return SparseMatrix(row, col, val)
