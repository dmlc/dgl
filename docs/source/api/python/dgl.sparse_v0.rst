.. _apibackend:

dgl.sparse
=================================

`dgl.sparse` is a library for sparse operators that are commonly used in GNN models.

Sparse matrix class
-------------------------
.. currentmodule:: dgl.sparse

.. class:: SparseMatrix

    Class for creating a sparse matrix representation

    There are a few ways to create a sparse matrix:

    * In COO format using row and col indices, use :func:`from_coo`.
    * In CSR format using row pointers and col indices, use :func:`from_csr`.
    * In CSC format using col pointers and row indices, use :func:`from_csc`.

    For example, one can create COO matrices as follows:

    Case1: Sparse matrix with row and column indices without values

        >>> row = torch.tensor([1, 1, 2])
        >>> col = torch.tensor([2, 4, 3])
        >>> A = from_coo(row, col)
        >>> A
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [2, 4, 3]]),
                     values=tensor([1., 1., 1.]),
                     shape=(3, 5), nnz=3)

    Case2: Sparse matrix with scalar/vector values

        >>> # vector values
        >>> val = torch.tensor([[1, 1], [2, 2], [3, 3]])
        >>> A = from_coo(row, col, val)
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
        >>> A = from_csr(indptr, indices, val)
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

    from_coo
    from_csr
    from_csc
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
    SparseMatrix.row
    SparseMatrix.col
    SparseMatrix.coo
    SparseMatrix.csr
    SparseMatrix.csc
    SparseMatrix.coalesce
    SparseMatrix.has_duplicate
    SparseMatrix.to_dense
    SparseMatrix.to
    SparseMatrix.cuda
    SparseMatrix.cpu
    SparseMatrix.float
    SparseMatrix.double
    SparseMatrix.int
    SparseMatrix.long
    SparseMatrix.transpose
    SparseMatrix.t
    SparseMatrix.T
    SparseMatrix.neg
    SparseMatrix.reduce
    SparseMatrix.sum
    SparseMatrix.smax
    SparseMatrix.smin
    SparseMatrix.smean
    SparseMatrix.softmax

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
    DiagMatrix.to_sparse
    DiagMatrix.to_dense
    DiagMatrix.to
    DiagMatrix.cuda
    DiagMatrix.cpu
    DiagMatrix.float
    DiagMatrix.double
    DiagMatrix.int
    DiagMatrix.long
    DiagMatrix.transpose
    DiagMatrix.t
    DiagMatrix.T
    DiagMatrix.neg
    DiagMatrix.inv

Operators
---------
.. currentmodule:: dgl.sparse

Elementwise Operators
````````

.. autosummary::
    :toctree: ../../generated/

    add
    sub
    mul
    div
    power

Matrix Multiplication
````````

.. autosummary::
    :toctree: ../../generated/

    matmul
    spmm
    bspmm
    spspmm
    sddmm
    bsddmm

Non-linear activation functions
````````

.. autosummary::
    :toctree: ../../generated/

    softmax
