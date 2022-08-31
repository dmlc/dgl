import torch
import operator
import numpy
from diag_matrix import DiagMatrix

def get_x_and_y(D1, D2):
    x = D1.val if isinstance(D1, DiagMatrix) else D1
    y = D2.val if isinstance(D2, DiagMatrix) else D2
    return x, y

def add(D1, D2):
    """Elementwise addition.

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix
        Diagonal matrix

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D1 = DiagMatrix(torch.arange(3))
    >>> D2 = DiagMatrix(torch.arange(3))
    >>> D1 + D2
    tensor([0, 2, 4])
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, 'The shape of diagonal matrix D1 {} and' \
        'D2 {} must match'.format(D1.shape, D2.shape)
        return DiagMatrix(D1.val + D2.val)
    raise RuntimeError('Elementwise add between diagonal and dense matrix is not supported.')

def sub(D1, D2):
    """Elementwise subtraction.

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix
        Diagonal matrix

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    ...
    >>> D1 - D2
    tensor([[0, 0, 0]])
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, 'The shape of diagonal matrix D1 {} and' \
        'D2 {} must match'.format(D1.shape, D2.shape)
        return DiagMatrix(D1.val - D2.val)
    raise RuntimeError('Elementwise sub between diagonal and dense matrix is not supported.')

def rsub(D1, D2):
    """Elementwise subtraction.

    Parameters
    ----------
    D1 : scalar
        scalar value
    D2 : DiagMatrix
        Diagonal matrix
    """
    raise RuntimeError('Elementwise sub between diagonal and dense matrix is not supported.')

def mul(D1, D2):
    """Elementwise multiplication.

    Parameters
    ----------
    D1 : DiagMatrix or scalar
        Diagonal matrix or scalar value
    D2 : DiagMatrix or scalar
        Diagonal matrix or scalar value.

   Returns
    -------
    DiagMatrix
        diagonal matrix

    Examples
    --------
    ...
    >>> D1 * D2
    tensor([0, 1, 4])
    >>> v_scalar = 2.5
    >>> D1 * v_scalar
    tensor([0.0000, 2.5000, 5.0000])
    >>> v_scalar * D1
    tensor([0.0000, 2.5000, 5.0000])
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, 'The shape of diagonal matrix D1 {} and' \
        'D2 {} must match'.format(D1.shape, D2.shape)
        return DiagMatrix(D1.val * D2.val)
    return DiagMatrix(D1.val * D2)

def div(D1, D2):
    """Elementwise division.

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or scalar
        Diagonal matrix or scalar value.

    Returns
    -------
    DiagMatrix
        diagonal matrix

    Examples
    --------
    >>> D1 = DiagMatrix(torch.arange(1, 4))
    >>> D2 = DiagMatrix(torch.arange(1, 4))
    >>> D1 / D2
    tensor([1., 1., 1.])
    >>> v_scalar = 2.5
    >>> D1 / v_scalar
    tensor([0.4000, 0.8000, 1.2000])
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, 'The shape of diagonal matrix D1 {} and' \
        'D2 {} must match'.format(D1.shape, D2.shape)
        return DiagMatrix(D1.val / D2.val)
    return DiagMatrix(D1.val / D2)

def rdiv(D1, D2):
    """Elementwise division.

    Parameters
    ----------
    D1 : scalar
        scalar value
    D2 : DiagMatrix
        Diagonal matrix
    """
    raise RuntimeError('Elementwise div between diagonal and dense matrix is not supported.')


def power(D1, D2):
    """Elementwise power operation.

    Parameters
    ----------
    D1 : DiagMatrix
        Diagonal matrix
    D2 : DiagMatrix or scalar
        Diagonal matrix or scalar value.

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> D1 = DiagMatrix(torch.arange(1, 4))
    >>> pow(D1, v_scalar)
    tensor([ 1.0000,  5.6569, 15.5885])
    """
    if isinstance(D1, DiagMatrix) and isinstance(D2, DiagMatrix):
        assert D1.shape == D2.shape, 'The shape of diagonal matrix D1 {} and' \
        'D2 {} must match'.format(D1.shape, D2.shape)
        return DiagMatrix(pow(D1.val, D2.val))
    return DiagMatrix(pow(D1.val, D2))

def rpower(D1, D2):
    """Elementwise power operator.

    Parameters
    ----------
    D1 : scalar
        scalar value
    D2 : DiagMatrix
        Diagonal matrix
    """
    raise RuntimeError('power operation between diagonal and dense matrix is not supported.')


DiagMatrix.__add__ = add
DiagMatrix.__radd__ = add
DiagMatrix.__sub__ = sub
DiagMatrix.__rsub__ = rsub
DiagMatrix.__mul__ = mul
DiagMatrix.__rmul__ = mul
DiagMatrix.__truediv__ = div
DiagMatrix.__rtruediv__ = rdiv
DiagMatrix.__pow__ = power
DiagMatrix.__rpow__ = rpower

if __name__ == '__main__':
    def test():
        w1 = torch.randn(5)
        w2 = torch.randn(5)
        D1 = DiagMatrix(torch.arange(1, 4))
        D2 = DiagMatrix(torch.arange(1, 4))
        v_scalar = 2.5

        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        for op in ops:

            # ***** diag matrix <op> diag matrix  ******
            comp_torch = "D1.val" + op + " D2.val"
            comp_spMat = "D1 " + op + " D2"
            print("Computing", comp_spMat)
            assert torch.equal(eval(comp_torch), eval(comp_spMat).val)

            # ***** diag matrix <op> scalar ******
            if op in ['*', '/']:
                comp_spMat = "D1 " + op + " v_scalar"
                print("Computing", comp_spMat)
                out = eval(comp_spMat)
                comp_torch = "D1.val" + op + " v_scalar"
                assert torch.equal(eval(comp_torch), out.val)

            # # ***** scalar <op> diag matrix ******
            if op in ['*']:
                comp_spMat =  "v_scalar " + op + " D1 "
                print("Computing", comp_spMat)
                out = eval(comp_spMat)
                comp_torch = "v_scalar" + op + " D1.val"
                assert torch.equal(eval(comp_torch), out.val)
            print()

        # Power  operator - pow(diag matrix, scalar)
        print("Computing pow(D1, v_scalar)")
        assert torch.equal(pow(D1.val, v_scalar), pow(D1, v_scalar).val)

    test()
    print('-------------------')
