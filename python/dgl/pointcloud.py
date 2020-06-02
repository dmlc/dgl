"""Python interfaces to DGL random number generators."""
import numpy as np

from ._ffi.function import _init_api
from . import backend as F
from . import ndarray as nd

def fps(data, batch_size, sample_points):
    assert F.shape(data)[0] >= sample_points
    res = _CAPI_FarthestPointSampler(F.zerocopy_to_dgl_ndarray(data),
                                     batch_size, sample_points)
    return F.zerocopy_from_dgl_ndarray(res)

_init_api('dgl.pointcloud', __name__)
