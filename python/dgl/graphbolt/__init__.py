"""Graphbolt."""
import os
import sys

import torch

### FROM DGL @todo
from .._ffi import libinfo


def load_graphbolt():
    """Load Graphbolt C++ library"""
    vers = torch.__version__.split("+", maxsplit=1)[0]

    if sys.platform.startswith("linux"):
        basename = f"libgraphbolt_pytorch_{vers}.so"
    elif sys.platform.startswith("darwin"):
        basename = f"libgraphbolt_pytorch_{vers}.dylib"
    elif sys.platform.startswith("win"):
        basename = f"graphbolt_pytorch_{vers}.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)

    dirname = os.path.dirname(libinfo.find_lib_path()[0])
    path = os.path.join(dirname, "graphbolt", basename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find DGL C++ graphbolt library at {path}"
        )

    try:
        torch.classes.load_library(path)
    except Exception:  # pylint: disable=W0703
        raise ImportError("Cannot load Graphbolt C++ library")


load_graphbolt()

# pylint: disable=wrong-import-position
from .base import *
from .minibatch import *
from .dataloader import *
from .dataset import *
from .feature_fetcher import *
from .feature_store import *
from .impl import *
from .itemset import *
from .item_sampler import *
from .minibatch_transformer import *
from .internal_utils import *
from .negative_sampler import *
from .sampled_subgraph import *
from .subgraph_sampler import *
from .external_utils import add_reverse_edges, exclude_seed_edges
from .internal import (
    compact_csc_format,
    numpy_save_aligned,
    unique_and_compact,
    unique_and_compact_csc_formats,
)

if torch.cuda.is_available() and not built_with_cuda():
    raise ImportError(
        "torch was installed with CUDA support while GraphBolt's CPU version "
        "is installed. Consider reinstalling GraphBolt with CUDA support, see "
        "installation instructions at https://www.dgl.ai/pages/start.html"
    )
