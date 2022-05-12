import numpy as np
import scipy.sparse as ssp
import pytest
import dgl
from compute.utils import parametrize_dtype
import backend as F
import unittest
import jax
import jax.numpy as jnp

def _random_simple_graph(idtype, dtype, ctx, M, N, max_nnz, srctype, dsttype, etype):
    src = np.random.randint(0, M, (max_nnz,))
    dst = np.random.randint(0, N, (max_nnz,))
    val = np.random.randn(max_nnz)
    a = ssp.csr_matrix((val, (src, dst)), shape=(M, N))
    a.sum_duplicates()
    a = a.tocoo()
    # shuffle edges
    perm = np.random.permutation(a.nnz)
    row = a.row[perm]
    col = a.col[perm]
    val = a.data[perm]
    a = ssp.csr_matrix((val, (row, col)), shape=(M, N))

    A = dgl.heterograph(
        {(srctype, etype, dsttype): (
            F.copy_to(F.tensor(row, dtype=idtype), ctx),
            F.copy_to(F.tensor(col, dtype=idtype), ctx))},
        num_nodes_dict={srctype: a.shape[0], dsttype: a.shape[1]})
    A.edata['w'] = F.copy_to(F.tensor(val, dtype=dtype), ctx)
    return a, A

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
@pytest.mark.parametrize('num_vtypes', [1, 2])
def test_csrmm_backward(idtype, dtype, num_vtypes):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 4, 3, 6, 'B', 'A' if num_vtypes == 1 else 'C', 'BA')
    A_row, A_col = A.edges(order='eid')
    B_row, B_col = B.edges(order='eid')
    A_row = F.asnumpy(A_row)
    A_col = F.asnumpy(A_col)
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    a_dense = F.attach_grad(F.tensor(a.todense(), dtype=dtype))
    b_dense = F.attach_grad(F.tensor(b.todense(), dtype=dtype))

    aw, bw = F.clone(A.edata['w']), F.clone(B.edata['w'])



    def f0(aw, bw):
        A.edata['w'], B.edata['w'] = aw, bw
        C = dgl.adj_product_graph(A, B, 'w')
        return F.reduce_sum(C.edata['w'])

    def f1(a_dense, b_dense):
        c_dense = F.matmul(a_dense, b_dense)
        return F.reduce_sum(c_dense)

    df0_daw, df0_dbw = jax.grad(f0, argnums=(0, 1))(aw, bw)
    df1_dadense, df1_dbdense = jax.grad(f1, argnums=(0, 1))(a_dense, b_dense)
    df1_dadense = df1_dadense[A_row, A_col]
    df1_dbdense = df1_dbdense[B_row, B_col]
    assert F.allclose(df0_daw, df1_dadense)
    assert F.allclose(df0_dbw, df1_dbdense)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
@pytest.mark.parametrize('nelems', [1, 2])
@unittest.skipIf(dgl.backend.backend_name=="jax", reason="JAX backward semantics is different.")
def test_csrsum_backward(idtype, dtype, nelems):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    A_row, A_col = A.edges(order='eid')
    B_row, B_col = B.edges(order='eid')
    A_row = F.asnumpy(A_row)
    A_col = F.asnumpy(A_col)
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    a_dense = F.attach_grad(F.tensor(a.todense(), dtype=dtype))
    b_dense = F.attach_grad(F.tensor(b.todense(), dtype=dtype))

    A.edata['w'] = F.attach_grad(A.edata['w'])
    B.edata['w'] = F.attach_grad(B.edata['w'])

    with F.record_grad():
        if nelems == 2:
            # Test for two element case
            C = dgl.adj_sum_graph([A, B], 'w')
            assert C.canonical_etypes == A.canonical_etypes
            C_dense = np.zeros((3, 4))
            C_row, C_col = C.edges(order='eid')
            C_row = F.asnumpy(C_row)
            C_col = F.asnumpy(C_col)
            C_dense[C_row, C_col] = F.asnumpy(C.edata['w'])
            c_dense = a_dense + b_dense
            assert np.allclose(C_dense, F.asnumpy(c_dense), rtol=1e-4, atol=1e-4)

            F.backward(F.reduce_sum(C.edata['w']) + F.reduce_sum(c_dense))
            a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
            b_dense_grad = F.asnumpy(F.grad(b_dense))[B_row, B_col]
            A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
            B_spspmm_grad = F.asnumpy(F.grad(B.edata['w']))
            assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)
            assert np.allclose(b_dense_grad, B_spspmm_grad, rtol=1e-4, atol=1e-4)
        elif nelems == 1:
            # Test for single element case
            C = dgl.adj_sum_graph([A], 'w')
            assert C.canonical_etypes == A.canonical_etypes
            C_dense = np.zeros((3, 4))
            C_row, C_col = C.edges(order='eid')
            C_row = F.asnumpy(C_row)
            C_col = F.asnumpy(C_col)
            C_dense[C_row, C_col] = F.asnumpy(C.edata['w'])
            c_dense = a_dense
            assert np.allclose(C_dense, F.asnumpy(c_dense), rtol=1e-4, atol=1e-4)

            F.backward(F.reduce_sum(C.edata['w']) + F.reduce_sum(c_dense))
            a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
            A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
            assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
@unittest.skipIf(dgl.backend.backend_name=="jax", reason="JAX backward semantics is different.")
def test_csrmask_backward(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    A_row, A_col = A.edges(order='eid')
    B_row, B_col = B.edges(order='eid')
    A_row = F.asnumpy(A_row)
    A_col = F.asnumpy(A_col)
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    a_dense = F.attach_grad(F.tensor(a.todense(), dtype=dtype))

    A.edata['w'] = F.attach_grad(A.edata['w'])

    with F.record_grad():
        # Test for two element case
        C1 = F.csrmask(A._graph, A.edata['w'], B._graph)
        if dgl.backend.backend_name == 'tensorflow':
            import tensorflow as tf
            C2 = tf.gather_nd(a_dense, tf.stack([B_row, B_col], 1))
        else:
            C2 = a_dense[B_row, B_col]
        assert F.allclose(C1, C2, rtol=1e-4, atol=1e-4)

        F.backward(F.reduce_sum(C1) + F.reduce_sum(C2))
        a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
        A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
        assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    test_csrmm_backward(F.int32, F.float32, 1)
    test_csrmm_backward(F.int64, F.float32, 1)
    test_csrmm_backward(F.int32, F.float32, 2)
    test_csrmm_backward(F.int64, F.float32, 2)
    test_csrsum_backward(F.int32, F.float32, 1)
    test_csrsum_backward(F.int64, F.float32, 1)
    test_csrsum_backward(F.int32, F.float32, 2)
    test_csrsum_backward(F.int64, F.float32, 2)
    test_csrmask_backward(F.int32, F.float32)
    test_csrmask_backward(F.int64, F.float32)
