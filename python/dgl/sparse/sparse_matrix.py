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
        >>> A = dglsp.spmatrix(indices)
        >>> A.coo()
        (tensor([1, 2, 1]), tensor([2, 4, 3]))
        """
        return self.c_sparse_matrix.coo()

    def indices(self) -> torch.Tensor:
        r"""Returns the coordinate list (COO) representation in one tensor with
        shape ``(2, nnz)``.

        See `COO in Wikipedia <https://en.wikipedia.org/wiki/
        Sparse_matrix#Coordinate_list_(COO)>`_.

        Returns
        -------
        torch.Tensor
            Stacked COO tensor with shape ``(2, nnz)``.

        Examples
        --------

        >>> indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
        >>> A = dglsp.spmatrix(indices)
        >>> A.indices()
        tensor([[1, 2, 1],
                [2, 4, 3]])
        """
        return self.c_sparse_matrix.indices()

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
        >>> A = dglsp.spmatrix(indices)
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
        >>> A = dglsp.spmatrix(indices)
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

    def is_diag(self):
        """Returns whether the sparse matrix is a diagonal matrix."""
        return self.c_sparse_matrix.is_diag()

    def index_select(self, dim: int, index: torch.Tensor):
        """Returns a sub-matrix selected according to the given index.

        Parameters
        ----------
        dim : int
            The dim to select from matrix, should be 0 or 1. `dim = 0` for
            rowwise selection and `dim = 1` for columnwise selection.
        index : torch.Tensor
            The selection index indicates which IDs from the `dim` should
            be chosen from the matrix.
            Note that duplicated ids are allowed.

        The function does not support autograd.

        Returns
        -------
        SparseMatrix
            The sub-matrix which contains selected rows or columns.

        Examples
        --------

        >>> indices = torch.tensor([0, 1, 1, 2, 3, 4], [0, 2, 4, 3, 5, 0]])
        >>> val = torch.tensor([0, 1, 2, 3, 4, 5])
        >>> A = dglsp.spmatrix(indices, val)

        Case 1: Select rows by IDs.

        >>> row_ids = torch.tensor([0, 1, 4])
        >>> A.index_select(0, row_ids)
        SparseMatrix(indices=tensor([[0, 1, 1, 2],
                                     [0, 2, 4, 0]]),
                     values=tensor([0, 1, 2, 5]),
                     shape=(3, 6), nnz=4)

        Case 2: Select columns by IDs.

        >>> column_ids = torch.tensor([0, 4, 5])
        >>> A.index_select(1, column_ids)
        SparseMatrix(indices=tensor([[0, 4, 1, 3],
                                     [0, 0, 1, 2]]),
                     values=tensor([0, 5, 2, 4]),
                     shape=(5, 3), nnz=4)
        """
        if dim not in (0, 1):
            raise ValueError("The selection dimension should be 0 or 1.")
        if isinstance(index, torch.Tensor):
            return SparseMatrix(self.c_sparse_matrix.index_select(dim, index))
        raise TypeError(f"{type(index).__name__} is unsupported input type.")

    def range_select(self, dim: int, index: slice):
        """Returns a sub-matrix selected according to the given range index.

        Parameters
        ----------
        dim : int
            The dim to select from matrix, should be 0 or 1. `dim = 0` for
            rowwise selection and `dim = 1` for columnwise selection.
        index : slice
            The selection slice indicates ID index from the `dim` should
            be chosen from the matrix.

        The function does not support autograd.

        Returns
        -------
        SparseMatrix
            The sub-matrix which contains selected rows or columns.

        Examples
        --------

        >>> indices = torch.tensor([0, 1, 1, 2, 3, 4], [0, 2, 4, 3, 5, 0]])
        >>> val = torch.tensor([0, 1, 2, 3, 4, 5])
        >>> A = dglsp.spmatrix(indices, val)

        Case 1: Select rows with given slice object.

        >>> A.range_select(0, slice(1, 3))
        SparseMatrix(indices=tensor([[0, 0, 1],
                                     [2, 4, 3]]),
                     values=tensor([1, 2, 3]),
                     shape=(2, 6), nnz=3)

        Case 2: Select columns with given slice object.

        >>> A.range_select(1, slice(3, 6))
        SparseMatrix(indices=tensor([[2, 1, 3],
                                     [0, 1, 2]]),
                     values=tensor([3, 2, 4]),
                     shape=(5, 3), nnz=3)
        """
        if dim not in (0, 1):
            raise ValueError("The selection dimension should be 0 or 1.")
        if isinstance(index, slice):
            if index.step not in (None, 1):
                raise NotImplementedError(
                    "Slice with step other than 1 are not supported yet."
                )
            start = 0 if index.start is None else index.start
            end = index.stop
            return SparseMatrix(
                self.c_sparse_matrix.range_select(dim, start, end)
            )
        raise TypeError(f"{type(index).__name__} is unsupported input type.")

    def sample(
        self,
        dim: int,
        fanout: int,
        ids: Optional[torch.Tensor] = None,
        replace: Optional[bool] = False,
        bias: Optional[bool] = False,
    ):
        """Returns a sampled matrix on the given dimension and sample arguments.

        Parameters
        ----------
        dim : int
            The dimension for sampling, should be 0 or 1. `dim = 0` for
            rowwise selection and `dim = 1` for columnwise selection.
        fanout : int
            The number of elements to randomly sample on each row or column.
        ids : torch.Tensor, optional
            An optional tensor containing row or column IDs from which to
            sample elements.
            NOTE: If `ids` is not provided (i.e., `ids = None`), the function
            will sample from all rows or columns.
        replace : bool, optional
            Indicates whether repeated sampling of the same element is allowed.
            When `replace = True`, repeated sampling is permitted; when
            `replace = False`, it is not allowed.
            NOTE: If `replace = False` and there are fewer elements than
            `fanout`, all non-zero elements will be sampled.
        bias : bool, optional
            A boolean flag indicating whether to enable biasing during sampling.
            When `bias = True`, the values of the sparse matrix will be used as
            bias weights.

        The function does not support autograd.

        Returns
        -------
        SparseMatrix
            A submatrix with the same shape as the original matrix, containing
            the randomly sampled non-zero elements.

        Examples
        --------

        >>> indices = torch.tensor([[0, 0, 1, 1, 2, 2, 2],
                                    [0, 2, 0, 1, 0, 1, 2]])
        >>> val = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        >>> A = dglsp.spmatrix(indices, val)

        Case 1: Sample rows with the given number and disable repeated sampling.

        >>> row_ids = torch.tensor([0, 2])
        >>> A.sample(0, 2, row_ids)
        SparseMatrix(indices=tensor([[0, 0, 1, 1],
                                     [0, 2, 0, 2]]),
                     values=tensor([0, 1, 4, 6]),
                     shape=(2, 3), nnz=4)

        Case 2: Sample cols with the given number and disable repeated sampling.

        >>> col_ids = torch.tensor([0, 2])
        >>> A.sample(1, 2, col_ids)
        SparseMatrix(indices=tensor([[0, 1, 0, 2],
                                     [0, 0, 1, 1]]),
                     values=tensor([0, 2, 1, 6]),
                     shape=(3, 2), nnz=4)

        Case 3: Sample rows with the given number and enable repeated sampling.

        >>> row_ids = torch.tensor([0, 1])
        >>> A.sample(0, 2, row_ids, True)
        SparseMatrix(indices=tensor([[0, 0, 1, 1],
                                     [0, 2, 0, 0]]),
                     values=tensor([0, 1, 2, 2]),
                     shape=(2, 3), nnz=3)

        Case 4: Sample cols with the given number and enable repeated sampling.

        >>> col_ids = torch.tensor([0, 1])
        >>> A.sample(1, 2, col_ids, True)
        SparseMatrix(indices=tensor([[0, 1, 1, 1],
                                     [0, 0, 1, 1]]),
                     values=tensor([0, 2, 3, 3]),
                     shape=(3, 2), nnz=3)
        """
        if ids is None:
            dim_size = self.shape[0] if dim == 0 else self.shape[1]
            ids = torch.range(
                0, dim_size, dtype=torch.int64, device=self.device
            )
        return SparseMatrix(
            self.c_sparse_matrix.sample(dim, fanout, ids, replace, bias)
        )

    def compact(
        self,
        dim: int,
        leading_indices: Optional[torch.Tensor] = None,
    ):
        """Compact sparse matrix by removing rows or columns without non-zero
        elements in the sparse matrix and relabeling indices of the dimension.

        This function serves a dual purpose: it allows you to reorganize the
        indices within a specific dimension (rows or columns) of the sparse
        matrix and, if needed, place certain 'leading_indices' at the beginning
        of the relabeled dimension.

        In the absence of 'leading_indices' (when it's set to `None`), the order
        of relabeled indices remains the same as the original order, except that
        rows or columns without non-zero elements are removed. When
        'leading_indices' are provided, they are positioned at the start of the
        relabeled dimension. To be precise, all rows selected by the specified
        indices will be remapped from 0 to length(indices) - 1. Rows that are not
        selected and contain any non-zero elements will be positioned after those
        remapped rows while maintaining their original order.

        This function mimics 'dgl.to_block', a method used to compress a sampled
        subgraph by eliminating redundant nodes. The 'leading_indices' parameter
        replicates the behavior of 'include_dst_in_src' in 'dgl.to_block',
        adding destination node information for message passing.
        Setting 'leading_indices' to column IDs when relabeling the row
        dimension, for example, achieves the same effect as including destination
        nodes in source nodes.

        Parameters
        ----------
        dim : int
            The dimension to relabel. Should be 0 or 1. Use `dim = 0` for rowwise
            relabeling and `dim = 1` for columnwise relabeling.
        leading_indices : torch.Tensor, optional
            An optional tensor containing row or column ids that should be placed
            at the beginning of the relabeled dimension.

        Returns
        -------
        Tuple[SparseMatrix, torch.Tensor]
            A tuple containing the relabeled sparse matrix and the index mapping
            of the relabeled dimension from the new index to the original index.

        Examples
        --------
        >>> indices = torch.tensor([[0, 2],
                                    [1, 2]])
        >>> A = dglsp.spmatrix(indices)
        >>> print(A.to_dense())
        tensor([[0., 1., 0.],
                [0., 0., 0.],
                [0., 0., 1.]])

        Case 1: Compact rows without indices.

        >>> B, original_rows = A.compact(dim=0, leading_indices=None)
        >>> print(B.to_dense())
        tensor([[0., 1., 0.],
                [0., 0., 1.]])
        >>> print(original_rows)
        torch.Tensor([0, 2])

        Case 2: Compact rows with indices.

        >>> B, original_rows = A.compact(dim=0, leading_indices=[1, 2])
        >>> print(B.to_dense())
        tensor([[0., 0., 0.],
                [0., 0., 1.],
                [0., 1., 0.],])
        >>> print(original_rows)
        torch.Tensor([1, 2, 0])
        """
        mat, idx = torch.ops.dgl_sparse.compact(
            self.c_sparse_matrix, dim, leading_indices
        )
        return SparseMatrix(mat), idx


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
    if shape is None:
        shape = (
            torch.max(indices[0]).item() + 1,
            torch.max(indices[1]).item() + 1,
        )
    if val is None:
        val = torch.ones(indices.shape[1]).to(indices.device)

    assert (
        val.dim() <= 2
    ), "The values of a SparseMatrix can only be scalars or vectors."
    return SparseMatrix(torch.ops.dgl_sparse.from_coo(indices, val, shape))


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
    assert row.shape[0] == col.shape[0]
    return spmatrix(torch.stack([row, col]), val, shape)


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


def diag(
    val: torch.Tensor, shape: Optional[Tuple[int, int]] = None
) -> SparseMatrix:
    """Creates a sparse matrix based on the diagonal values.

    Parameters
    ----------
    val : torch.Tensor
        Diagonal of the matrix, in shape ``(N)`` or ``(N, D)``
    shape : tuple[int, int], optional
        If specified, :attr:`len(val)` must be equal to :attr:`min(shape)`,
        otherwise, it will be inferred from :attr:`val`, i.e., ``(N, N)``

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: 5-by-5 diagonal matrix with scaler values on the diagonal

    >>> import torch
    >>> val = torch.ones(5)
    >>> dglsp.diag(val)
    SparseMatrix(indices=tensor([[0, 1, 2, 3, 4],
                                 [0, 1, 2, 3, 4]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(5, 5), nnz=5)

    Case2: 5-by-10 diagonal matrix with scaler values on the diagonal

    >>> val = torch.ones(5)
    >>> dglsp.diag(val, shape=(5, 10))
    SparseMatrix(indices=tensor([[0, 1, 2, 3, 4],
                                 [0, 1, 2, 3, 4]]),
                 values=tensor([1., 1., 1., 1., 1.]),
                 shape=(5, 10), nnz=5)

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
    len_val = len(val)
    if shape is not None:
        assert len_val == min(shape), (
            f"Expect len(val) to be min(shape) for a diagonal matrix, got"
            f"{len_val} for len(val) and {shape} for shape."
        )
    else:
        shape = (len_val, len_val)
    return SparseMatrix(torch.ops.dgl_sparse.from_diag(val, shape))


def identity(
    shape: Tuple[int, int],
    d: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> SparseMatrix:
    r"""Creates a sparse matrix with ones on the diagonal and zeros elsewhere.

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
    SparseMatrix
        Sparse matrix

    Examples
    --------

    Case1: 3-by-3 matrix with scaler diagonal values

    .. code::

        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]

    >>> dglsp.identity(shape=(3, 3))
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 3), nnz=3)

    Case2: 3-by-5 matrix with scaler diagonal values

    .. code::

        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0]]

    >>> dglsp.identity(shape=(3, 5))
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)

    Case3: 3-by-3 matrix with vector diagonal values

    >>> dglsp.identity(shape=(3, 3), d=2)
    SparseMatrix(indices=tensor([[0, 1, 2],
                                 [0, 1, 2]]),
                 values=tensor([[1., 1.],
                                [1., 1.],
                                [1., 1.]]),
                 shape=(3, 3), nnz=3, val_size=(2,))
    """
    len_val = min(shape)
    if d is None:
        val_shape = (len_val,)
    else:
        val_shape = (len_val, d)
    val = torch.ones(val_shape, dtype=dtype, device=device)
    return diag(val, shape)


def from_torch_sparse(torch_sparse_tensor: torch.Tensor) -> SparseMatrix:
    """Creates a sparse matrix from a torch sparse tensor, which can have coo,
    csr, or csc layout.

    Parameters
    ----------
    torch_sparse_tensor : torch.Tensor
        Torch sparse tensor

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> val = torch.ones(3)
    >>> torch_coo = torch.sparse_coo_tensor(indices, val)
    >>> dglsp.from_torch_sparse(torch_coo)
    SparseMatrix(indices=tensor([[1, 1, 2],
                                 [2, 4, 3]]),
                 values=tensor([1., 1., 1.]),
                 shape=(3, 5), nnz=3)
    """
    assert torch_sparse_tensor.layout in (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
    ), (
        f"Cannot convert Pytorch sparse tensor with layout "
        f"{torch_sparse_tensor.layout} to DGL sparse."
    )
    if torch_sparse_tensor.layout == torch.sparse_coo:
        # Use ._indices() and ._values() to access uncoalesced indices and
        # values.
        return spmatrix(
            torch_sparse_tensor._indices(),
            torch_sparse_tensor._values(),
            torch_sparse_tensor.shape[:2],
        )
    elif torch_sparse_tensor.layout == torch.sparse_csr:
        return from_csr(
            torch_sparse_tensor.crow_indices(),
            torch_sparse_tensor.col_indices(),
            torch_sparse_tensor.values(),
            torch_sparse_tensor.shape[:2],
        )
    else:
        return from_csc(
            torch_sparse_tensor.ccol_indices(),
            torch_sparse_tensor.row_indices(),
            torch_sparse_tensor.values(),
            torch_sparse_tensor.shape[:2],
        )


def to_torch_sparse_coo(spmat: SparseMatrix) -> torch.Tensor:
    """Creates a torch sparse coo tensor from a sparse matrix.

    Parameters
    ----------
    spmat : SparseMatrix
        Sparse matrix

    Returns
    -------
    torch.Tensor
        torch tensor with torch.sparse_coo layout

    Examples
    --------

    >>> indices = torch.tensor([[1, 1, 2], [2, 4, 3]])
    >>> val = torch.ones(3)
    >>> spmat = dglsp.spmatrix(indices, val)
    >>> dglsp.to_torch_sparse_coo(spmat)
    tensor(indices=tensor([[1, 1, 2],
                           [2, 4, 3]]),
           values=tensor([1., 1., 1.]),
           size=(3, 5), nnz=3, layout=torch.sparse_coo)
    """
    shape = spmat.shape
    if spmat.val.dim() > 1:
        shape += spmat.val.shape[1:]
    return torch.sparse_coo_tensor(spmat.indices(), spmat.val, shape)


def to_torch_sparse_csr(spmat: SparseMatrix) -> torch.Tensor:
    """Creates a torch sparse csr tensor from a sparse matrix.

    Note that converting a sparse matrix to torch csr tensor could change the
    order of non-zero values.

    Parameters
    ----------
    spmat : SparseMatrix
        Sparse matrix

    Returns
    -------
    torch.Tensor
        Torch tensor with torch.sparse_csr layout

    Examples
    --------

    >>> indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
    >>> val = torch.arange(3)
    >>> spmat = dglsp.spmatrix(indices, val)
    >>> dglsp.to_torch_sparse_csr(spmat)
    tensor(crow_indices=tensor([0, 0, 2, 3]),
           col_indices=tensor([2, 3, 4]),
           values=tensor([0, 2, 1]), size=(3, 5), nnz=3,
           layout=torch.sparse_csr)
    """
    shape = spmat.shape
    if spmat.val.dim() > 1:
        shape += spmat.val.shape[1:]
    indptr, indices, value_indices = spmat.csr()
    val = spmat.val
    if value_indices is not None:
        val = val[value_indices]
    return torch.sparse_csr_tensor(indptr, indices, val, shape)


def to_torch_sparse_csc(spmat: SparseMatrix) -> torch.Tensor:
    """Creates a torch sparse csc tensor from a sparse matrix.

    Note that converting a sparse matrix to torch csc tensor could change the
    order of non-zero values.

    Parameters
    ----------
    spmat : SparseMatrix
        Sparse matrix

    Returns
    -------
    torch.Tensor
        Torch tensor with torch.sparse_csc layout

    Examples
    --------

    >>> indices = torch.tensor([[1, 2, 1], [2, 4, 3]])
    >>> val = torch.arange(3)
    >>> spmat = dglsp.spmatrix(indices, val)
    >>> dglsp.to_torch_sparse_csc(spmat)
    tensor(ccol_indices=tensor([0, 0, 0, 1, 2, 3]),
           row_indices=tensor([1, 1, 2]),
           values=tensor([0, 2, 1]), size=(3, 5), nnz=3,
           layout=torch.sparse_csc)
    """
    shape = spmat.shape
    if spmat.val.dim() > 1:
        shape += spmat.val.shape[1:]
    indptr, indices, value_indices = spmat.csc()
    val = spmat.val
    if value_indices is not None:
        val = val[value_indices]
    return torch.sparse_csc_tensor(indptr, indices, val, shape)


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
