import torch
import operator
from sp_matrix import SparseMatrix

def check_sparsity(A, B):
    if not torch.equal(A.indices("COO"), B.indices("COO")):
        raise ValueError('The two input matrices have different sparsity pattern')

def add(A, B):
    """Elementwise addition.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    >>> row = torch.tensor([1, 1, 2])
    >>> col = torch.tensor([2, 4, 3])
    >>> w1 = torch.tensor([3 ,4, 5])
    >>> w2 = torch.tensor([-1, -3, -3])
    >>> C = A + A
    >>> C.val
    tensor([ 6,  8, 10])
    >>> C = A(w2) + A(w1)
    tensor([2, 1, 2])
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, 'The shape of sparse matrix A {} and' \
        'B {} must match'.format(A.shape, B.shape)
        print("shape", A.shape, B.shape)
        check_sparsity(A, B)
    else:
        raise RuntimeError('Elementwise add between sparse and dense matrix is not supported.')
    return SparseMatrix(A.row, A.col, A.val + B.val)

def sub(A, B):
    """Elementwise subtraction.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix
        Sparse matrix

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    ...
    >>> C = A - A
    >>> C.val
    tensor([0, 0, 0])
    >>> C = A(w2) - A(w1)
    tensor([-4, -7, -8])
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        assert A.shape == B.shape, 'The shape of sparse matrix A {} and' \
        'B {} must match'.format(A.shape, B.shape)
        print("shape", A.shape, B.shape)
        check_sparsity(A, B)
    else:
        raise RuntimeError('Elementwise sub between sparse and dense matrix is not supported.')
    return SparseMatrix(A.row, A.col, A.val - B.val)

def rsub(A, B):
    """Elementwise subtraction.

    Parameters
    ----------
    A : scalar
        scalar value
    B : SparseMatrix
        Sparse matrix
    """
    raise RuntimeError('Elementwise sub between sparse and dense matrix is not supported.')

def mul(A, B):
    """Elementwise multiplication.

    Parameters
    ----------
    A : SparseMatrix or scalar
        Sparse matrix or scalar value
    B : SparseMatrix or scalar
        Sparse matrix or scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    ...
    >>> C = A * A
    >>> C.val
    tensor([ 9, 16, 25])
    >>> C = A(w2) * A(w1)
    tensor([ -3, -12, -15])
    >>> v_scalar = 2.5
    >>> C = A * v_scalar
    >>> C.val
    tensor([ 7.5000, 10.0000, 12.5000])
    >>> C = v_scalar * A
    >>> C.val
    tensor([ 7.5000, 10.0000, 12.5000])
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        check_sparsity(A, B)
    a_val = A.val if isinstance(A, SparseMatrix) else A
    b_val = B.val if isinstance(B, SparseMatrix) else B
    return SparseMatrix(A.row, A.col, a_val * b_val)

def div(A, B):
    """Elementwise division.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : SparseMatrix or scalar
        Sparse matrix or scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    ...
    >>> C = A / A
    >>> C.val
    tensor([1., 1., 1.])
    >>> C = A(w2) / A(w1)
    tensor([-0.3333, -0.7500, -0.6000])
    >>> v_scalar = 2.5
    >>> C = A / v_scalar
    >>> C.val
    tensor([1.2000, 1.6000, 2.0000])
    """
    if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):
        check_sparsity(A, B)
    a_val = A.val if isinstance(A, SparseMatrix) else A
    b_val = B.val if isinstance(B, SparseMatrix) else B
    return SparseMatrix(A.row, A.col, a_val / b_val)

def rdiv(A, B):
    """Elementwise division.

    Parameters
    ----------
    A : scalar
        scalar value
    B : SparseMatrix
        Sparse matrix
    """
    raise RuntimeError('Elementwise div between sparse and dense matrix is not supported.')

def power(A, B):
    """Elementwise power operation.

    Parameters
    ----------
    A : SparseMatrix
        Sparse matrix
    B : scalar
        scalar value.

    Returns
    -------
    SparseMatrix
        Sparse matrix

    Examples
    --------
    ...
    >>> C = pow(A, v_scalar)
    >>> C.val
    tensor([15.5885, 32.0000, 55.9017])
    """
    if isinstance(B, SparseMatrix):
        raise RuntimeError('power operation between two sparse matrices is not supported')
    return SparseMatrix(A.row, A.col, torch.pow(A.val, B))

def rpower(A, B):
    """Elementwise subtraction.

    Parameters
    ----------
    A : scalar
        scalar value.
    B : SparseMatrix
        Sparse matrix.
    """
    raise RuntimeError('power operation between sparse and dense matrix is not supported.')


SparseMatrix.__add__ = add
SparseMatrix.__radd__ = add
SparseMatrix.__sub__ = sub
SparseMatrix.__rsub__ = rsub
SparseMatrix.__mul__ = mul
SparseMatrix.__rmul__ = mul
SparseMatrix.__truediv__ = div
SparseMatrix.__rtruediv__ = rdiv
SparseMatrix.__pow__ = power
SparseMatrix.__rpow__ = rpower

if __name__ == '__main__':
    def test():
        row = torch.tensor([1, 1, 2])
        col = torch.tensor([2, 4, 3])
        w1 = torch.tensor([3 ,4, 5])
        w2 = torch.tensor([-1, -3, -3])
        # w1 = torch.tensor([[3, 4], [5, 6], [7, 8]])
        # w2 = torch.tensor([[1, 1], [2, 2], [-1, -1]])
        A = SparseMatrix(row, col, w1)
        v_scalar = 2.5

        # Binary op (sparse_matrix <op> sparse_matrix)
        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        for op in ops:
            # ***** sparse_matrix <op> sparse matrix ******
            comp_torch = "w1 " + op + " w1"
            comp_torch2 = "w2 " + op + " w1"
            comp_spMat = "A " + op + " A"
            comp_spMat2 = "A(w2) " + op + " A(w1)"
            print("Computing", comp_spMat)  # A + A
            assert torch.equal(eval(comp_torch), eval(comp_spMat).val)
            print("Computing", comp_spMat2)  # A(w1) + A(w2)
            assert torch.equal(eval(comp_torch2), eval(comp_spMat2).val)

        # ***** sparse_matrix <op> scalar (supports '*', '/' and pow op) ******
        print("Computing A * v_scalar")
        assert torch.equal(w1 * v_scalar, (A * v_scalar).val)
        print("Computing A / v_scalar")
        assert torch.equal(w1 / v_scalar, (A / v_scalar).val)
        print("Computing pow(A, v_scalar)")
        assert torch.equal(pow(w1, v_scalar), pow(A, v_scalar).val)

        # ***** scalar <op> sparse_matrix (supports only '*' op)******
        print("Computing v_scalar * A")
        assert torch.equal(v_scalar * w1, (v_scalar * A).val)
        print()

    test()
    print('-------------------')
