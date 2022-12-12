"""dgl sparse class."""
import os
import sys

import torch

from .._ffi import libinfo
from .diag_matrix import *
from .elementwise_op import *
from .sparse_matrix import *
from .unary_op_diag import *
from .unary_op_sp import *


def load_dgl_sparse():
    """Load DGL C++ sparse library"""
    version = torch.__version__.split("+", maxsplit=1)[0]
    basename = f"libdgl_sparse_pytorch_{version}.so"
    dirname = os.path.dirname(libinfo.find_lib_path()[0])
    path = os.path.join(dirname, "dgl_sparse", basename)

    try:
        torch.classes.load_library(path)
    except Exception:  # pylint: disable=W0703
        raise ImportError("Cannot load DGL C++ sparse library")


# TODO(zhenkun): support other platforms
if sys.platform.startswith("linux"):
    load_dgl_sparse()
