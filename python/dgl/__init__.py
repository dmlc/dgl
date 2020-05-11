"""DGL root package."""
# Windows compatibility
# This initializes Winsock and performs cleanup at termination as required
import socket

# Need to ensure that the backend framework is imported before load dgl libs,
# otherwise weird cuda problem happens
from .backend import load_backend, backend_name

from . import function
from . import contrib
from . import container
from . import random
from . import sampling

from ._ffi.runtime_ctypes import TypeCode
from ._ffi.function import register_func, get_global_func, list_global_func_names, extract_ext_funcs
from ._ffi.base import DGLError, __version__

from .base import ALL, NTYPE, NID, ETYPE, EID
from .readout import *
from .batched_heterograph import *
from .convert import *
from .graph import DGLGraph, batch, unbatch
from .generators import *
from .heterograph import DGLHeteroGraph
from .nodeflow import *
from .traversal import *
from .transform import *
from .propagate import *
from .udf import NodeBatch, EdgeBatch
