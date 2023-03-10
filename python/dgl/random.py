"""Python interfaces to DGL random number generators."""
import numpy as np

from . import backend as F, ndarray as nd
from ._ffi.function import _init_api

__all__ = ["seed"]


def seed(val):
    """Set the random seed of DGL.

    Parameters
    ----------
    val : int
        The seed.
    """
    _CAPI_SetSeed(val)


def choice(a, size, replace=True, prob=None):  # pylint: disable=invalid-name
    """An equivalent to :func:`numpy.random.choice`.

    Use this function if you:

    * Perform a non-uniform sampling (probability tensor is given).
    * Sample a small set from a very large population (ratio <5%) uniformly
      *without* replacement.
    * Have a backend tensor on hand and does not want to convert it to numpy
      back and forth.

    Compared to :func:`numpy.random.choice`, it is slower when replace is True
    and is comparable when replace is False. It wins when the population is
    very large and the number of draws are quite small (e.g., draw <5%). The
    reasons are two folds:

    * When ``a`` is a large integer, it avoids creating a large range array as
      numpy does.
    * When draw ratio is small, it switches to a hashmap based implementation.

    It out-performs numpy for non-uniform sampling in general cases.

    Parameters
    ----------
    a : 1-D tensor or int
        If an ndarray, a random sample is generated from its elements. If an int,
        the random sample is generated as if a were F.arange(a)
    size : int or tuple of ints
        Output shape. E.g., for size ``(m, n, k)``, then ``m * n * k`` samples are drawn.
    replace : bool, optional
        If true, sample with replacement.
    prob : 1-D tensor, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all entries in a.

    Returns
    -------
    samples : 1-D tensor
        The generated random samples
    """
    # TODO(minjie): support RNG as one of the arguments.
    if isinstance(size, tuple):
        num = np.prod(size)
    else:
        num = size

    if F.is_tensor(a):
        population = F.shape(a)[0]
    else:
        population = a

    if prob is None:
        prob = nd.NULL["int64"]
    else:
        prob = F.zerocopy_to_dgl_ndarray(prob)

    bits = 64  # index array is in 64-bit
    chosen_idx = _CAPI_Choice(
        int(num), int(population), prob, bool(replace), bits
    )
    chosen_idx = F.zerocopy_from_dgl_ndarray(chosen_idx)

    if F.is_tensor(a):
        chosen = F.gather_row(a, chosen_idx)
    else:
        chosen = chosen_idx

    if isinstance(size, tuple):
        return F.reshape(chosen, size)
    else:
        return chosen


_init_api("dgl.rng", __name__)
