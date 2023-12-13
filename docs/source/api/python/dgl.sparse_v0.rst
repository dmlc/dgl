.. _apibackend:

dgl.sparse
=================================

`dgl.sparse` is a library for sparse operators that are commonly used in GNN models.

Sparse matrix class
-------------------------
.. currentmodule:: dgl.sparse

.. class:: SparseMatrix

    A SparseMatrix can be created from Coordinate format indices using the
    :func:`spmatrix` constructor:

        >>> indices = torch.tensor([[1, 1, 2],
        >>>                         [2, 4, 3]])
        >>> A = dglsp.spmatrix(indices)
        SparseMatrix(indices=tensor([[1, 1, 2],
                                     [2, 4, 3]]),
                     values=tensor([1., 1., 1.]),
                     shape=(3, 5), nnz=3)

Creation Ops
````````

.. autosummary::
    :toctree: ../../generated/

    spmatrix
    val_like
    from_coo
    from_csr
    from_csc
    diag
    identity

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
    SparseMatrix.indices
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

Broadcast operators
````````

.. autosummary::
    :toctree: ../../generated/

    sp_broadcast_v
    sp_add_v
    sp_sub_v
    sp_mul_v
    sp_div_v