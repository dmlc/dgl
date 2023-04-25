"""dgl sparse class."""
import os
import sys

import torch

from .._ffi import libinfo
from .broadcast import *
from .elementwise_op import *
from .elementwise_op_sp import *
from .matmul import *
from .reduction import *  # pylint: disable=W0622
from .sddmm import *
from .softmax import *
from .sparse_matrix import *
from .unary_op import *


def load_dgl_sparse():
    """Load DGL C++ sparse library"""
    version = torch.__version__.split("+", maxsplit=1)[0]

    if sys.platform.startswith("linux"):
        basename = f"libdgl_sparse_pytorch_{version}.so"
    elif sys.platform.startswith("darwin"):
        basename = f"libdgl_sparse_pytorch_{version}.dylib"
    elif sys.platform.startswith("win"):
        basename = f"dgl_sparse_pytorch_{version}.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)

    dirname = os.path.dirname(libinfo.find_lib_path()[0])
    path = os.path.join(dirname, "dgl_sparse", basename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find DGL C++ sparse library at {path}")

    try:
        torch.classes.load_library(path)
    except Exception:  # pylint: disable=W0703
        raise ImportError("Cannot load DGL C++ sparse library")


load_dgl_sparse()
