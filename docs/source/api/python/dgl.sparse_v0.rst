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
	:members: shape, nnz, dtype, device, row, col, val, __call__, indices, coo, csr, csc, dense, t, T, transpose,
            reduce, sum, smax, smin, smean, __neg__, inv, softmax, __matmul__

.. autosummary::
    :toctree: ../../generated/

    create_from_coo
    create_from_csr
    create_from_csc

Diagonal matrix class
-------------------------
.. currentmodule:: dgl.mock_sparse

.. autoclass:: DiagMatrix
	:members: val, shape, __call__, nnz, dtype, device, as_sparse, t, T, transpose,
            reduce, sum, smax, smin, smean, __neg__, inv, softmax, __matmul__

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
