"""Python interfaces to DGL random number generators."""
import numpy as np

from ._ffi.function import _init_api
from . import backend as F
from . import ndarray as nd

def fps(data, batch_ptr, npoints):
    assert F.shape(data)[0] >= npoints
    res = _CAPI_FarthestPointSampler(F.zerocopy_to_dgl_ndarray(data),
                                     F.zerocopy_to_dgl_ndarray(batch_ptr),
                                     npoints)
    return F.zerocopy_from_dgl_ndarray(res)

_init_api('dgl.fps', __name__)
