"""
The ``dgl`` package contains data structure for storing structural and feature data
(i.e., the :class:`DGLGraph` class) and also utilities for generating, manipulating
and transforming graphs.
"""


# Windows compatibility
# This initializes Winsock and performs cleanup at termination as required
import socket

# Backend and logging should be imported before other modules.
from .logging import enable_verbose_logging  # usort: skip
from .backend import backend_name, load_backend  # usort: skip

from . import (
    container,
    cuda,
    dataloading,
    function,
    ops,
    random,
    sampling,
    storages,
)
from ._ffi.base import __version__, DGLError
from ._ffi.function import (
    extract_ext_funcs,
    get_global_func,
    list_global_func_names,
    register_func,
)

from ._ffi.runtime_ctypes import TypeCode

from .base import ALL, EID, ETYPE, NID, NTYPE
from .readout import *
from .batch import *
from .convert import *
from .generators import *
from .dataloading import (
    set_dst_lazy_features,
    set_edge_lazy_features,
    set_node_lazy_features,
    set_src_lazy_features,
)
from .heterograph import (  # pylint: disable=reimported
    DGLGraph,
    DGLGraph as DGLHeteroGraph,
)
from .merge import *
from .subgraph import *
from .traversal import *
from .transforms import *
from .propagate import *
from .random import *
from . import optim
from .data.utils import load_graphs, save_graphs
from .frame import LazyFeature
from .global_config import is_libxsmm_enabled, use_libxsmm
from .utils import apply_each
from .mpops import *
from .homophily import *
from .label_informativeness import *
