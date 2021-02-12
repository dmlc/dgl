"""Utilities for tf NN package"""
# pylint: disable=no-member, invalid-name
from tensorflow.keras import layers  # pylint: disable=W0235
import tensorflow as tf


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
    A : tf.Tensor
        lhs tensor
    B : tf.Tensor
        rhs tensor

    Returns
    -------
    C : tf.Tensor
        result tensor
    """
    if A.dtype == tf.int64 and len(A.shape) == 1:
        return tf.gather(B, A)
    else:
        return tf.matmul(A, B)


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
    A : tf.Tensor
        lhs tensor
    B : tf.Tensor
        rhs tensor
    index : tf.Tensor
        index tensor

    Returns
    -------
    C : tf.Tensor
        return tensor
    """
    if A.dtype == tf.int64 and len(A.shape) == 1:
        # following is a faster version of B[index, A, :]
        B = tf.reshape(B, (-1, B.shape[2]))
        flatidx = index * B.shape[1] + A
        return tf.gather(B, flatidx)
    else:
        BB = tf.gather(B, index)
        return tf.squeeze(tf.matmul(tf.expand_dims(A, 1), BB), 1)


class Identity(layers.Layer):
    """A placeholder identity operator that is argument-insensitive.
    """

    def call(self, x):
        """Return input"""
        return x
