from .sampler import NeighborSampler, LayerSampler
from .randomwalk import *
from .dis_sampler import SamplerSender, SamplerReceiver
from .dis_sampler import SamplerPool
from ..._ffi.function import _init_api

def seed(val):
    """Set the seed of randomized methods in DGL.

    The randomized methods include various samplers and random walk routines.

    Parameters
    ----------
    val : int
        The seed
    """
    _CAPI_SetSeed(val)

_init_api('dgl.rng', __name__)
