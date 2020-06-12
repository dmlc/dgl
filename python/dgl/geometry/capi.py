"""Python interfaces to DGL farthest point sampler."""
from .._ffi.function import _init_api
from .. import backend as F

def farthest_point_sampler(data, batch_size, sample_points, dist, start_idx, result):
    """Farthest Point Sampler

    Parameters
    ----------
    data : tensor
        A tensor of shape (N, d) where N is the number of points and d is the dimension.
    batch_size : int
        The number of batches in the ``data``. N should be divisible by batch_size.
    sample_points : int
        The number of points to sample in each batch.
    dist : tensor
        Pre-allocated tensor of shape (N, ) for to-sample distance.
    start_idx : tensor of int
        Pre-allocated tensor of shape (batch_size, ) for the starting sample in each batch.
    result : tensor of int
        Pre-allocated tensor of shape (sample_points * batch_size, ) for the sampled index.

    Returns
    -------
    No return value. The input variable ``result`` will be overwriten with sampled indices.

    """
    assert F.shape(data)[0] >= sample_points * batch_size
    assert F.shape(data)[0] % batch_size == 0

    _CAPI_FarthestPointSampler(F.zerocopy_to_dgl_ndarray(data),
                               batch_size, sample_points,
                               F.zerocopy_to_dgl_ndarray(dist),
                               F.zerocopy_to_dgl_ndarray(start_idx),
                               F.zerocopy_to_dgl_ndarray(result))

_init_api('dgl.geometry', __name__)
