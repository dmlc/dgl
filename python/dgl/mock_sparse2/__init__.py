"""dgl sparse class."""
import sys
import os
import torch

from .._ffi import libinfo
from .sparse_matrix import *
from .diag_matrix import *
from .elementwise_op import *

def load_dgl_sparse():
    """Load DGL C++ sparse library"""
    version = torch.__version__.split("+", maxsplit=1)[0]
    # TODO(zhenkun): support other platforms
    assert sys.platform.startswith("linux")
    basename = f"libdgl_sparse_pytorch_{version}.so"
    dirname = os.path.dirname(libinfo.find_lib_path()[0])
    path = os.path.join(dirname, "dgl_sparse", basename)

    try:
        torch.classes.load_library(path)
    except Exception:  # pylint: disable=W0703
        raise ImportError("Cannot load DGL C++ sparse library")


load_dgl_sparse()
