.. _apibackend:

dgl.sparse
=================================

`dgl.sparse` is a library for sparse operators that are commonly used in GNN models.

.. warning::
    This is an experimental package. The sparse operators provided in this library do not guarantee the same performance as their message-passing api counterparts.

Sparse matrix class
-------------------------
.. currentmodule:: dgl.sparse

.. class:: SparseMatrix

    Class for creating a sparse matrix representation

    There are a few ways to create a sparse matrix:

    * In COO format using row and col indices, use :func:`create_from_coo`.
    * In CSR format using row pointers and col indices, use :func:`create_from_csr`.
    * In CSC format using col pointers and row indices, use :func:`create_from_csc`.

    For example, one can create COO matrices as follows:

    Case1: Sparse matrix with row and column indices without values

        >>> row = torch.tensor([1, 1, 2])
        >>> col = torch.tensor([2, 4, 3])
        >>> A = create_from_coo(row, col)
        >>> A
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [2, 4, 3]]),
                     values=tensor([1., 1., 1.]),
                     shape=(3, 5), nnz=3)

    Case2: Sparse matrix with scalar/vector values

        >>> # vector values
        >>> val = torch.tensor([[1, 1], [2, 2], [3, 3]])
        >>> A = create_from_coo(row, col, val)
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [2, 4, 3]]),
                     values=tensor([[1, 1],
                                    [2, 2],
                                    [3, 3]]),
                     shape=(3, 5), nnz=3)

    Similarly, one can create a CSR matrix as follows:

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

Creators
````````

.. autosummary::
    :toctree: ../../generated/

    create_from_coo
    create_from_csr
    create_from_csc
    val_like

Attributes and methods
``````````````````````

.. autosummary::
    :toctree: ../../generated/

    SparseMatrix.shape
    SparseMatrix.nnz
    SparseMatrix.dtype
    SparseMatrix.device
    SparseMatrix.val
    SparseMatrix.__repr__
    SparseMatrix.row
    SparseMatrix.col
    SparseMatrix.indices
    SparseMatrix.coo
    SparseMatrix.csr
    SparseMatrix.csc
    SparseMatrix.coalesce
    SparseMatrix.has_duplicate
    SparseMatrix.dense
    SparseMatrix.t
    SparseMatrix.T
    SparseMatrix.transpose
    SparseMatrix.reduce
    SparseMatrix.sum
    SparseMatrix.smax
    SparseMatrix.smin
    SparseMatrix.smean
    SparseMatrix.neg
    SparseMatrix.softmax
    SparseMatrix.__matmul__

Diagonal matrix class
-------------------------
.. currentmodule:: dgl.sparse

.. class:: DiagMatrix

Creators
````````

.. autosummary::
    :toctree: ../../generated/

    diag
    identity

Attributes and methods
``````````````````````

.. autosummary::
    :toctree: ../../generated/

    DiagMatrix.shape
    DiagMatrix.nnz
    DiagMatrix.dtype
    DiagMatrix.device
    DiagMatrix.val
    DiagMatrix.__repr__
    DiagMatrix.as_sparse
    DiagMatrix.dense
    DiagMatrix.t
    DiagMatrix.T
    DiagMatrix.transpose
    DiagMatrix.neg
    DiagMatrix.inv
    DiagMatrix.__matmul__

Operators
---------
.. currentmodule:: dgl.sparse

.. autosummary::
    :toctree: ../../generated/

    sp_add
    sp_mul
    sp_power
    diag_add
    diag_sub
    diag_mul
    diag_div
    diag_power
    add
    power
    spmm
    bspmm
    spspmm
    mm
    sddmm
    bsddmm
    softmax
