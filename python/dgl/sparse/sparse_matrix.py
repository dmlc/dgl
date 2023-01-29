"""DGL sparse matrix module."""
# pylint: disable= invalid-name
from typing import Optional, Tuple

import torch


class SparseMatrix:
    r"""Class for sparse matrix."""

    def __init__(self, c_sparse_matrix: torch.ScriptObject):
        self.c_sparse_matrix = c_sparse_matrix

    def __repr__(self):
        return _sparse_matrix_str(self)

    @property
    def val(self) -> torch.Tensor:
        """Returns the values of the non-zero elements.

        Returns
        -------
        torch.Tensor
            Values of the non-zero elements
        """
        return self.c_sparse_matrix.val()

    @property
    def shape(self) -> Tuple[int]:
        """Returns the shape of the sparse matrix.

        Returns
        -------
        Tuple[int]
            The shape of the sparse matrix
        """
        return tuple(self.c_sparse_matrix.shape())

    @property
    def nnz(self) -> int:
        """Returns the number of non-zero elements in the sparse matrix.

        Returns
        -------
        int
            The number of non-zero elements of the matrix
        """
        return self.c_sparse_matrix.nnz()

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the sparse matrix.

        Returns
        -------
        torch.dtype
            Data type of the sparse matrix
        """
        return self.c_sparse_matrix.val().dtype

    @property
    def device(self) -> torch.device:
        """Returns the device the sparse matrix is on.

        Returns
        -------
        torch.device
            The device the sparse matrix is on
        """
        return self.c_sparse_matrix.device()

    @property
    def row(self) -> torch.Tensor:
        """Returns the row indices of the non-zero elements.

        Returns
        -------
        torch.Tensor
            Row indices of the non-zero elements
        """
        return self.coo()[0]

    @property
    def col(self) -> torch.Tensor:
        """Returns the column indices of the non-zero elements.

        Returns
        -------
        torch.Tensor
            Column indices of the non-zero elements
        """
        return self.coo()[1]

    def coo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the coordinate list (COO) representation of the sparse
        matrix.

        See `COO in Wikipedia <https://en.wikipedia.org/wiki/
        Sparse_matrix#Coordinate_list_(COO)>`_.

        Returns
        -------
        torch.Tensor
            Row coordinate
        torch.Tensor
            Column coordinate

        Examples
        --------

        >>> indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
        >>> A = from_coo(dst, src)
        >>> A.coo()
        (tensor([1, 2, 1]), tensor([2, 4, 3]))
        """
        return self.c_sparse_matrix.coo()

    def csr(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Returns the compressed sparse row (CSR) representation of the sparse
        matrix.

        See `CSR in Wikipedia <https://en.wikipedia.org/wiki/
        Sparse_matrix#Compressed_sparse_row_(CSR, _CRS_or_Yale_format)>`_.

        This function also returns value indices as an index tensor, indicating
        the order of the values of non-zero elements in the CSR representation.
        A ``None`` value indices array indicates the order of the values stays
        the same as the values of the SparseMatrix.

        Returns
        -------
        torch.Tensor
            Row indptr
        torch.Tensor
            Column indices
        torch.Tensor
            Value indices

        Examples
        --------

        >>> indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
        >>> A = from_coo(dst, src)
        >>> A.csr()
        (tensor([0, 0, 2, 3]), tensor([2, 3, 4]), tensor([0, 2, 1]))
        """
        return self.c_sparse_matrix.csr()

    def csc(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Returns the compressed sparse column (CSC) representation of the
        sparse matrix.

        See `CSC in Wikipedia <https://en.wikipedia.org/wiki/
        Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>`_.

        This function also returns value indices as an index tensor, indicating
        the order of the values of non-zero elements in the CSC representation.
        A ``None`` value indices array indicates the order of the values stays
        the same as the values of the SparseMatrix.

        Returns
        -------
        torch.Tensor
            Column indptr
        torch.Tensor
            Row indices
        torch.Tensor
            Value indices

        Examples
        --------

        >>> indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
        >>> A = from_coo(dst, src)
        >>> A.csc()
        (tensor([0, 0, 0, 1, 2, 3]), tensor([1, 1, 2]), tensor([0, 2, 1]))
        """
        return self.c_sparse_matrix.csc()

    def to_dense(self) -> torch.Tensor:
        """Returns a copy in dense matrix format of the sparse matrix.

        Returns
        -------
        torch.Tensor
            The copy in dense matrix format
        """
        row, col = self.coo()
        val = self.val
        shape = self.shape + val.shape[1:]
        mat = torch.zeros(shape, device=self.device, dtype=self.dtype)
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
        """Returns the transpose of this sparse matrix.

        Returns
        -------
        SparseMatrix
            The transpose of this sparse matrix.

        Examples
        --------

        >>> indices = torch.tensor([[1, 1, 3], [2, 1, 3]])
        >>> val = torch.tensor([1, 1, 2])
        >>> A = dglsp.spmatrix(indices, val)
        >>> A = A.transpose()
        SparseMatrix(indices=tensor([[2, 1, 3],
                                     [1, 1, 3]]),
                     values=tensor([1, 1, 2]),
                     shape=(4, 4), nnz=3)
        """
        return SparseMatrix(self.c_sparse_matrix.transpose())

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
        SparseMatrix
            The converted matrix

        Examples
        --------

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]])
        >>> A = dglsp.spmatrix(indices, shape=(3, 4))
        >>> A.to(device="cuda:0", dtype=torch.int32)
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]], device='cuda:0'),
                     values=tensor([1, 1, 1], device='cuda:0',
                                   dtype=torch.int32),
                     shape=(3, 4), nnz=3)
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        if device == self.device and dtype == self.dtype:
            return self
        elif device == self.device:
            return val_like(self, self.val.to(dtype=dtype))
        else:
            # TODO(#5119): Find a better moving strategy instead of always
            # convert to COO format.
            row, col = self.coo()
            row = row.to(device=device)
            col = col.to(device=device)
            val = self.val.to(device=device, dtype=dtype)
            return from_coo(row, col, val, self.shape)

    def cuda(self):
        """Moves the matrix to GPU. If the matrix is already on GPU, the
        original matrix will be returned. If multiple GPU devices exist,
        ``cuda:0`` will be selected.

        Returns
        -------
        SparseMatrix
            The matrix on GPU

        Examples
        --------

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]])
        >>> A = dglsp.spmatrix(indices, shape=(3, 4))
        >>> A.cuda()
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]], device='cuda:0'),
                     values=tensor([1., 1., 1.], device='cuda:0'),
                     shape=(3, 4), nnz=3)
        """
        return self.to(device="cuda")

    def cpu(self):
        """Moves the matrix to CPU. If the matrix is already on CPU, the
        original matrix will be returned.

        Returns
        -------
        SparseMatrix
            The matrix on CPU

        Examples
        --------

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]]).to("cuda")
        >>> A = dglsp.spmatrix(indices, shape=(3, 4))

        >>> A.cpu()
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]]),
                     values=tensor([1., 1., 1.]),
                     shape=(3, 4), nnz=3)
        """
        return self.to(device="cpu")

    def float(self):
        """Converts the matrix values to float32 data type. If the matrix
        already uses float data type, the original matrix will be returned.

        Returns
        -------
        SparseMatrix
            The matrix with float values

        Examples
        --------

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]])
        >>> val = torch.ones(len(row)).long()
        >>> A = dglsp.spmatrix(indices, val, shape=(3, 4))
        >>> A.float()
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]]),
                     values=tensor([1., 1., 1.]),
                     shape=(3, 4), nnz=3)
        """
        return self.to(dtype=torch.float)

    def double(self):
        """Converts the matrix values to double data type. If the matrix already
        uses double data type, the original matrix will be returned.

        Returns
        -------
        SparseMatrix
            The matrix with double values

        Examples
        --------

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]])
        >>> A = dglsp.spmatrix(indices, shape=(3, 4))
        >>> A.double()
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]]),
                     values=tensor([1., 1., 1.], dtype=torch.float64),
                     shape=(3, 4), nnz=3)
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

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]])
        >>> A = dglsp.spmatrix(indices, shape=(3, 4))
        >>> A.int()
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]]),
                     values=tensor([1, 1, 1], dtype=torch.int32),
                     shape=(3, 4), nnz=3)
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

        >>> indices = torch.tensor([[1, 1, 2], [1, 2, 0]])
        >>> A = dglsp.spmatrix(indices, shape=(3, 4))
        >>> A.long()
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [1, 2, 0]]),
                     values=tensor([1, 1, 1]),
                     shape=(3, 4), nnz=3)
        """
        return self.to(dtype=torch.long)

    def coalesce(self):
        """Returns a coalesced sparse matrix.

        A coalesced sparse matrix satisfies the following properties:

          - the indices of the non-zero elements are unique,
          - the indices are sorted in lexicographical order.

        The coalescing process will accumulate the non-zero elements of the same
        indices by summation.

        The function does not support autograd.

        Returns
        -------
        SparseMatrix
            The coalesced sparse matrix

        Examples
        --------
        >>> indices = torch.tensor([[1, 0, 0, 0, 1], [1, 1, 1, 2, 2]])
        >>> val = torch.tensor([0, 1, 2, 3, 4])
        >>> A = dglsp.spmatrix(indices, val)
        >>> A.coalesce()
        SparseMatrix(indices=tensor([[0, 0, 1, 1],
                                     [1, 2, 1, 2]]),
                     values=tensor([3, 3, 0, 4]),
                     shape=(2, 3), nnz=4)
        """
        return SparseMatrix(self.c_sparse_matrix.coalesce())

    def has_duplicate(self):
        """Returns ``True`` if the sparse matrix contains duplicate indices.

        Examples
        --------
        >>> indices = torch.tensor([[1, 0, 0, 0, 1], [1, 1, 1, 2, 2]])
        >>> val = torch.tensor([0, 1, 2, 3, 4])
        >>> A = dglsp.spmatrix(indices, val)
        >>> A.has_duplicate()
        True
        >>> A.coalesce().has_duplicate()
        False
        """
        return self.c_sparse_matrix.has_duplicate()


def spmatrix(
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    r"""Creates a sparse matrix from Coordinate format indices.

    Parameters
    ----------
    indices : tensor.Tensor
        The indices are the coordinates of the non-zero elements in the matrix,
        which should have shape of ``(2, N)`` where the first row is the row
        indices and the second row is the column indices of non-zero elements.
    val : tensor.Tensor, optional
        The values of shape ``(nnz)`` or ``(nnz, D)``. If None, it will be a
        tensor of shape ``(nnz)`` filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`row` and :attr:`col`,
        i.e., ``(row.max() + 1, col.max() + 1)``. Otherwise, :attr:`shape`
        should be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix with row and column indices without values.

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> A = dglsp.spmatrix(indices)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)
    >>> # Specify shape
    >>> A = dglsp.spmatrix(indices, shape=(5, 5))
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(5, 5), nnz=3)

    Case2: Sparse matrix with scalar values.

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> val = torch.tensor([[1.], [2.], [3.]])
    >>> A = dglsp.spmatrix(indices, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1.],
                                [2.],
                                [3.]]),
                 shape=(3, 5), nnz=3, val_size=(1,))

    Case3: Sparse matrix with vector values.

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> val = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
    >>> A = dglsp.spmatrix(indices, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1., 1.],
                                [2., 2.],
                                [3., 3.]]),
                 shape=(3, 5), nnz=3, val_size=(2,))
    """
    return from_coo(indices[0], indices[1], val, shape)


def from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    r"""Creates a sparse matrix from a coordinate list (COO), which stores a list
    of (row, column, value) tuples.

    See `COO in Wikipedia
    <https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)>`_.

    Parameters
    ----------
    row : torch.Tensor
        The row indices of shape ``(nnz)``
    col : torch.Tensor
        The column indices of shape ``(nnz)``
    val : torch.Tensor, optional
        The values of shape ``(nnz)`` or ``(nnz, D)``. If None, it will be a
        tensor of shape ``(nnz)`` filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`row` and :attr:`col`,
        i.e., ``(row.max() + 1, col.max() + 1)``. Otherwise, :attr:`shape`
        should be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix with row and column indices without values.

    >>> dst = torch.tensor([1, 1, 2])
    >>> src = torch.tensor([2, 4, 3])
    >>> A = dglsp.from_coo(dst, src)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)
    >>> # Specify shape
    >>> A = dglsp.from_coo(dst, src, shape=(5, 5))
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(5, 5), nnz=3)

    Case2: Sparse matrix with scalar values.

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> val = torch.tensor([[1.], [2.], [3.]])
    >>> A = dglsp.spmatrix(indices, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1.],
                                [2.],
                                [3.]]),
                 shape=(3, 5), nnz=3, val_size=(1,))

    Case3: Sparse matrix with vector values.

    >>> dst = torch.tensor([1, 1, 2])
    >>> src = torch.tensor([2, 4, 3])
    >>> val = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
    >>> A = dglsp.from_coo(dst, src, val)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([[1., 1.],
                                [2., 2.],
                                [3., 3.]]),
                 shape=(3, 5), nnz=3, val_size=(2,))
    """
    if shape is None:
        shape = (torch.max(row).item() + 1, torch.max(col).item() + 1)
    if val is None:
        val = torch.ones(row.shape[0]).to(row.device)

    assert (
        val.dim() <= 2
    ), "The values of a SparseMatrix can only be scalars or vectors."

    return SparseMatrix(torch.ops.dgl_sparse.from_coo(row, col, val, shape))


def from_csr(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    r"""Creates a sparse matrix from compress sparse row (CSR) format.

    See `CSR in Wikipedia <https://en.wikipedia.org/wiki/
    Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_.

    For row i of the sparse matrix

    - the column indices of the non-zero elements are stored in
      ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : torch.Tensor
        Pointer to the column indices of shape ``(N + 1)``, where ``N`` is the
        number of rows
    indices : torch.Tensor
        The column indices of shape ``(nnz)``
    val : torch.Tensor, optional
        The values of shape ``(nnz)`` or ``(nnz, D)``. If None, it will be a
        tensor of shape ``(nnz)`` filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`indptr` and
        :attr:`indices`, i.e., ``(len(indptr) - 1, indices.max() + 1)``.
        Otherwise, :attr:`shape` should be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix without values

    .. code::

        [[0, 1, 0],
         [0, 0, 1],
         [1, 1, 1]]

    >>> indptr = torch.tensor([0, 1, 2, 5])
    >>> indices = torch.tensor([1, 2, 0, 1, 2])
    >>> A = dglsp.from_csr(indptr, indices)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(3, 3), nnz=5)
    >>> # Specify shape
    >>> A = dglsp.from_csr(indptr, indices, shape=(3, 5))
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(3, 5), nnz=5)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> indptr = torch.tensor([0, 1, 2, 5])
    >>> indices = torch.tensor([1, 2, 0, 1, 2])
    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> A = dglsp.from_csr(indptr, indices, val)
    SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
                                 [1, 2, 0, 1, 2]]),
                 values=tensor([[1, 1],
                                [2, 2],
                                [3, 3],
                                [4, 4],
                                [5, 5]]),
                 shape=(3, 3), nnz=5, val_size=(2,))
    """
    if shape is None:
        shape = (indptr.shape[0] - 1, torch.max(indices) + 1)
    if val is None:
        val = torch.ones(indices.shape[0]).to(indptr.device)

    assert (
        val.dim() <= 2
    ), "The values of a SparseMatrix can only be scalars or vectors."

    return SparseMatrix(
        torch.ops.dgl_sparse.from_csr(indptr, indices, val, shape)
    )


def from_csc(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    val: Optional[torch.Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> SparseMatrix:
    r"""Creates a sparse matrix from compress sparse column (CSC) format.

    See `CSC in Wikipedia <https://en.wikipedia.org/wiki/
    Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>`_.

    For column i of the sparse matrix

    - the row indices of the non-zero elements are stored in
      ``indices[indptr[i]: indptr[i+1]]``
    - the corresponding values are stored in ``val[indptr[i]: indptr[i+1]]``

    Parameters
    ----------
    indptr : torch.Tensor
        Pointer to the row indices of shape N + 1, where N is the
        number of columns
    indices : torch.Tensor
        The row indices of shape nnz
    val : torch.Tensor, optional
        The values of shape ``(nnz)`` or ``(nnz, D)``. If None, it will be a
        tensor of shape ``(nnz)`` filled by 1.
    shape : tuple[int, int], optional
        If not specified, it will be inferred from :attr:`indptr` and
        :attr:`indices`, i.e., ``(indices.max() + 1, len(indptr) - 1)``.
        Otherwise, :attr:`shape` should be no smaller than this.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: Sparse matrix without values

    .. code::

        [[0, 1, 0],
         [0, 0, 1],
         [1, 1, 1]]

    >>> indptr = torch.tensor([0, 1, 3, 5])
    >>> indices = torch.tensor([2, 0, 2, 1, 2])
    >>> A = dglsp.from_csc(indptr, indices)
    SparseMatrix(indices=tensor([[2, 0, 2, 1, 2],
                                 [0, 1, 1, 2, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(3, 3), nnz=5)
    >>> # Specify shape
    >>> A = dglsp.from_csc(indptr, indices, shape=(5, 3))
    SparseMatrix(indices=tensor([[2, 0, 2, 1, 2],
                                 [0, 1, 1, 2, 2]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(5, 3), nnz=5)

    Case2: Sparse matrix with scalar/vector values. Following example is with
    vector data.

    >>> indptr = torch.tensor([0, 1, 3, 5])
    >>> indices = torch.tensor([2, 0, 2, 1, 2])
    >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> A = dglsp.from_csc(indptr, indices, val)
    SparseMatrix(indices=tensor([[2, 0, 2, 1, 2],
                                 [0, 1, 1, 2, 2]]),
                 values=tensor([[1, 1],
                                [2, 2],
                                [3, 3],
                                [4, 4],
                                [5, 5]]),
                 shape=(3, 3), nnz=5, val_size=(2,))
    """
    if shape is None:
        shape = (torch.max(indices) + 1, indptr.shape[0] - 1)
    if val is None:
        val = torch.ones(indices.shape[0]).to(indptr.device)

    assert (
        val.dim() <= 2
    ), "The values of a SparseMatrix can only be scalars or vectors."

    return SparseMatrix(
        torch.ops.dgl_sparse.from_csc(indptr, indices, val, shape)
    )


def val_like(mat: SparseMatrix, val: torch.Tensor) -> SparseMatrix:
    """Creates a sparse matrix from an existing sparse matrix using new values.

    The new sparse matrix will have the same non-zero indices as the given
    sparse matrix and use the given values as the new non-zero values.

    Parameters
    ----------
    mat : SparseMatrix
        An existing sparse matrix with non-zero values
    val : torch.Tensor
        The new values of the non-zero elements, a tensor of shape ``(nnz)`` or
        ``(nnz, D)``

    Returns
    -------
    SparseMatrix
        New sparse matrix

    Examples
    --------

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> val = torch.ones(3)
    >>> A = dglsp.spmatrix(indices, val)
    >>> A = dglsp.val_like(A, torch.tensor([2, 2, 2]))
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([2, 2, 2]),
                 shape=(3, 5), nnz=3)
    """
    assert (
        val.dim() <= 2
    ), "The values of a SparseMatrix can only be scalars or vectors."

    return SparseMatrix(torch.ops.dgl_sparse.val_like(mat.c_sparse_matrix, val))


def _sparse_matrix_str(spmat: SparseMatrix) -> str:
    """Internal function for converting a sparse matrix to string
    representation.
    """
    indices_str = str(torch.stack(spmat.coo()))
    values_str = str(spmat.val)
    meta_str = f"shape={spmat.shape}, nnz={spmat.nnz}"
    if spmat.val.dim() > 1:
        val_size = tuple(spmat.val.shape[1:])
        meta_str += f", val_size={val_size}"
    prefix = f"{type(spmat).__name__}("

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    final_str = (
        "indices="
        + _add_indent(indices_str, len("indices="))
        + ",\n"
        + "values="
        + _add_indent(values_str, len("values="))
        + ",\n"
        + meta_str
        + ")"
    )
    final_str = prefix + _add_indent(final_str, len(prefix))
    return final_str
