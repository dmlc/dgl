"""graphbolt."""
import os
import sys

import torch

from .._ffi import libinfo

from .data_fetcher import *
from .graph_storage import *
from .minibatch_sampler import *
from .negative_sampler import *
from .subgraph_sampler import *

def load_graphbolt():
    """Load Graphbolt C++ library"""
    version = torch.__version__.split("+", maxsplit=1)[0]

    if sys.platform.startswith("linux"):
        basename = f"libgraphbolt_pytorch_{version}.so"
    elif sys.platform.startswith("darwin"):
        basename = f"libgraphbolt_pytorch_{version}.dylib"
    elif sys.platform.startswith("win"):
        basename = f"graphbolt_pytorch_{version}.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)

    dirname = os.path.dirname('/home/ubuntu/workspace/dgl/graphbolt/build/')
    path = os.path.join(dirname, basename)
    dirname = os.path.dirname(libinfo.find_lib_path()[0])
    path = os.path.join(dirname, "dgl_sparse", basename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find DGL C++ sparse library at {path}")

    try:
        torch.classes.load_library(path)
    except Exception:  # pylint: disable=W0703
        raise ImportError("Cannot load Graphbolt C++ library")


load_graphbolt()



