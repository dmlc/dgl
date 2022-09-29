.. _apibackend:

dgl.mock_sparse
=================================

`dgl_sparse` is a library for sparse operators that are commonly used in GNN models.

.. warning::
    This is an experimental package. The sparse operators provided in this library do not guarantee the same performance as their message-passing api counterparts.

Sparse matrix class
-------------------------
.. currentmodule:: dgl.mock_sparse

.. autoclass:: SparseMatrix

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
