"""Utilities for pytorch NN package"""
#pylint: disable=no-member, invalid-name

from mxnet import nd
import numpy as np

def matmul_maybe_select(A, B):
    """Perform Matrix multiplication C = A * B but A could be an integer id vector.

    If A is an integer vector, we treat it as multiplying a one-hot encoded tensor.
    In this case, the expensive dense matrix multiply can be replaced by a much
    cheaper index lookup.

    For example,
    ::

        A = [2, 0, 1],
        B = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]

    then matmul_maybe_select(A, B) is equivalent to
    ::

        [[0, 0, 1],     [[0.1, 0.2],
         [1, 0, 0],  *   [0.3, 0.4],
         [0, 1, 0]]      [0.5, 0.6]]

    In all other cases, perform a normal matmul.

    Parameters
    ----------
    A : mxnet.NDArray
        lhs tensor
    B : mxnet.NDArray
        rhs tensor

    Returns
    -------
    C : mxnet.NDArray
        result tensor
    """
    if A.dtype in (np.int32, np.int64) and len(A.shape) == 1:
        return nd.take(B, A, axis=0)
    else:
        return nd.dot(A, B)

def bmm_maybe_select(A, B, index):
    """Slice submatrices of A by the given index and perform bmm.

    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)

    For case (1), C is computed by bmm:
    ::

        C[i, :] = matmul(A[i, :], B[index[i], :, :])

    For case (2), C is computed by index select:
    ::

        C[i, :] = B[index[i], A[i], :]

    Parameters
    ----------
    A : mxnet.NDArray
        lhs tensor
    B : mxnet.NDArray
        rhs tensor
    index : mxnet.NDArray
        index tensor

    Returns
    -------
    C : mxnet.NDArray
        return tensor
    """
    if A.dtype in (np.int32, np.int64) and len(A.shape) == 1:
        return B[index, A, :]
    else:
        BB = nd.take(B, index, axis=0)
        return nd.batch_dot(A.expand_dims(1), BB).squeeze()

def normalize(x, p=2, axis=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension
     :math:`1` for normalization.

    Args:
        x: input ndarray of any shape
        ord (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """
    denom = nd.clip(nd.norm(x, ord=p, axis=axis, keepdims=True), eps, float('inf'))
    return x / denom
