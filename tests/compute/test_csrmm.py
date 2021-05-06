import numpy as np
import scipy.sparse as ssp
import dgl
from utils import parametrize_dtype
import backend as F

def _random_simple_graph(idtype, dtype, ctx, M, N, max_nnz, srctype, dsttype, etype):
    src = np.random.randint(0, M, (max_nnz,))
    dst = np.random.randint(0, N, (max_nnz,))
    val = np.random.randn(max_nnz)
    a = ssp.csr_matrix((val, (src, dst)), shape=(M, N))
    a.sum_duplicates()
    a = a.tocoo()
    A = dgl.heterograph(
        {('A', 'AB', 'B'): (
            F.copy_to(F.tensor(a.row, dtype=idtype), ctx),
            F.copy_to(F.tensor(a.col, dtype=idtype), ctx))},
        num_nodes_dict={'A': a.shape[0], 'B': a.shape[1]})
    A.edata['w'] = F.copy_to(F.tensor(a.data, dtype=dtype), ctx)
    return a, A

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
def test_csrmm(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 600, 700, 9000, 'B', 'C', 'BC')
    C, C_weights = dgl.sparse.csrmm(A._graph, A.edata['w'], B._graph, B.edata['w'], 2)
    C_adj = C.adjacency_matrix_scipy(0, True, 'csr')
    C_adj.data = F.asnumpy(C_weights)
    C_adj = F.tensor(C_adj.todense(), dtype=dtype)
    c = F.tensor((a * b).todense(), dtype=dtype)
    assert F.allclose(C_adj, c)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
def test_csrsum(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    C, C_weights = dgl.sparse.csrsum([A._graph, B._graph], [A.edata['w'], B.edata['w']])
    C_adj = C.adjacency_matrix_scipy(0, True, 'csr')
    C_adj.data = F.asnumpy(C_weights)
    C_adj = F.tensor(C_adj.todense(), dtype=dtype)
    c = F.tensor((a + b).todense(), dtype=dtype)
    assert F.allclose(C_adj, c)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
def test_csrmask(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    C = dgl.sparse.csrmask(A._graph, A.edata['w'], B._graph)
    c = F.tensor(a.tocsr()[b != 0], dtype)
    assert F.allclose(C, c)

if __name__ == '__main__':
    test_csrmm(F.int32)
    test_csrmm(F.int64)
    test_csrsum(F.int32)
    test_csrsum(F.int64)
    test_csrmask(F.int32)
    test_csrmask(F.int64)
