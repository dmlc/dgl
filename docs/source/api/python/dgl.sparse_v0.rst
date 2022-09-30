.. _apibackend:

dgl.mock_sparse
=================================

`dgl_sparse` is a library for sparse operators that are commonly used in GNN models.

.. warning::
    This is an experimental package. The sparse operators provided in this library do not guarantee the same performance as their message-passing api counterparts.

Sparse matrix class
-------------------------
.. currentmodule:: dgl.mock_sparse

.. class:: SparseMatrix

    Class for creating a sparse matrix representation. The row and column indices of the sparse matrix can be the source
    (row) and destination (column) indices of a homogeneous or heterogeneous graph.

    There are a few ways to create a sparse matrix:

    * In COO format using row and col indices, use :func:`create_from_coo`.
    * In CSR format using row pointers and col indices, use :func:`create_from_csr`.
    * In CSC format using col pointers and row indices, use :func:`create_from_csc`.

    For example, we can create COO matrix as follows:

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

    Similarly, we can create CSR matrix as follows:

        >>> indptr = torch.tensor([0, 1, 2, 5])
        >>> indices = torch.tensor([1, 2, 0, 1, 2])
        >>> val = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        >>> A = create_from_csr(indptr, indices, val)
        >>> A
        SparseMatrix(indices=tensor([[0, 1, 2, 2, 2],
                [1, 2, 0, 1, 2]]),
        values=tensor([[1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]]),
        shape=(3, 3), nnz=5)

Sparse matrix class attributes
------------------------------

.. autosummary::

    SparseMatrix.shape
    SparseMatrix.nnz
    SparseMatrix.dtype
    SparseMatrix.device
    SparseMatrix.row
    SparseMatrix.col
    SparseMatrix.val
    __call__
    SparseMatrix.indices
    SparseMatrix.coo
    SparseMatrix.csr
    SparseMatrix.csc
    SparseMatrix.dense
    SparseMatrix.t
    SparseMatrix.T
    SparseMatrix.transpose
    SparseMatrix.reduce
    SparseMatrix.sum
    SparseMatrix.smax
    SparseMatrix.smin
    SparseMatrix.smean
    SparseMatrix.__neg__
    SparseMatrix.inv
    SparseMatrix.softmax
    SparseMatrix.__matmul__

.. autosummary::
    :toctree: ../../generated/

    create_from_coo
    create_from_csr
    create_from_csc

Diagonal matrix class
-------------------------
.. currentmodule:: dgl.mock_sparse

.. autoclass:: DiagMatrix

Diagonal matrix class attributes
--------------------------------

.. autosummary::

    DiagMatrix.val
    DiagMatrix.shape
    DiagMatrix.__call__
    DiagMatrix.nnz
    DiagMatrix.dtype
    DiagMatrix.device
    DiagMatrix.as_sparse
    DiagMatrix.t
    DiagMatrix.T
    DiagMatrix.transpose
    DiagMatrix.reduce
    DiagMatrix.sum
    DiagMatrix.smax
    DiagMatrix.smin
    DiagMatrix.smean
    DiagMatrix.__neg__
    DiagMatrix.inv
    DiagMatrix.softmax
    DiagMatrix.__matmul__

.. autosummary::
    :toctree: ../../generated/

    diag
    identity

Operators
---------
.. currentmodule:: dgl.mock_sparse

.. autosummary::
    :toctree: ../../generated/

    spmm
    spspmm
    bspmm
    bspspmm
    sddmm
