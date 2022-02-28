"""
The ``dgl`` package contains data structure for storing structural and feature data
(i.e., the :class:`DGLGraph` class) and also utilities for generating, manipulating
and transforming graphs.
"""


# Windows compatibility
# This initializes Winsock and performs cleanup at termination as required
import socket

# setup logging before everything
from .logging import enable_verbose_logging

# Should import backend before importing anything else
from .backend import load_backend, backend_name

from . import function
from . import contrib
from . import container
from . import distributed
from . import random
from . import sampling
from . import storages
from . import dataloading
from . import ops
from . import cuda
from . import _dataloading  # legacy dataloading modules

from ._ffi.runtime_ctypes import TypeCode
from ._ffi.function import register_func, get_global_func, list_global_func_names, extract_ext_funcs
from ._ffi.base import DGLError, __version__

from .base import ALL, NTYPE, NID, ETYPE, EID
from .readout import *
from .batch import *
from .convert import *
from .generators import *
from .heterograph import DGLHeteroGraph
from .heterograph import DGLHeteroGraph as DGLGraph  # pylint: disable=reimported
from .dataloading import set_src_lazy_features, set_dst_lazy_features, set_edge_lazy_features, \
    set_node_lazy_features
from .merge import *
from .subgraph import *
from .traversal import *
from .transforms import *
from .propagate import *
from .random import *
from .data.utils import save_graphs, load_graphs
from . import optim
from .frame import LazyFeature
from .utils import apply_each

from ._deprecate.graph import DGLGraph as DGLGraphStale
from ._deprecate.nodeflow import *
