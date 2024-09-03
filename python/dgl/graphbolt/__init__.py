"""Graphbolt."""
import os
import sys

from .internal_utils import *

CUDA_ALLOCATOR_ENV_WARNING_STR = """
An experimental feature for CUDA allocations is turned on for better allocation
pattern resulting in better memory usage for minibatch GNN training workloads.
See https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf,
and set the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
if you want to disable it and set it True to acknowledge and disable the warning.
"""
cuda_allocator_env = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
WARNING_STR_TO_BE_SHOWN = None
configs = (
    {}
    if cuda_allocator_env is None or len(cuda_allocator_env) == 0
    else {
        kv_pair.split(":")[0]: kv_pair.split(":")[1]
        for kv_pair in cuda_allocator_env.split(",")
    }
)
if "expandable_segments" in configs:
    if configs["expandable_segments"] != "True":
        WARNING_STR_TO_BE_SHOWN = (
            "You should consider `expandable_segments:True` in the"
            " environment variable `PYTORCH_CUDA_ALLOC_CONF` for lower"
            " memory usage. See "
            "https://pytorch.org/docs/stable/notes/cuda.html"
            "#optimizing-memory-usage-with-pytorch-cuda-alloc-conf"
        )
else:
    configs["expandable_segments"] = "True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
        [k + ":" + v for k, v in configs.items()]
    )
    WARNING_STR_TO_BE_SHOWN = CUDA_ALLOCATOR_ENV_WARNING_STR
del configs
del cuda_allocator_env
del CUDA_ALLOCATOR_ENV_WARNING_STR

# pylint: disable=wrong-import-position, wrong-import-order
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
            f"Unable to locate the DGL C++ GraphBolt library at {path}. This "
            "error typically occurs due to a version mismatch between the "
            "installed DGL and the PyTorch version you are currently using. "
            "Please ensure that your DGL installation is compatible with your "
            "PyTorch version. For more information, refer to the installation "
            "guide at https://www.dgl.ai/pages/start.html."
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
from .datapipes import *
from .dataset import *
from .feature_fetcher import *
from .feature_store import *
from .impl import *
from .itemset import *
from .item_sampler import *
from .minibatch_transformer import *
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

if torch.cuda.is_available() and WARNING_STR_TO_BE_SHOWN is not None:
    gb_warning(WARNING_STR_TO_BE_SHOWN)
del WARNING_STR_TO_BE_SHOWN

torch.ops.graphbolt.set_num_io_uring_threads(
    min((torch.get_num_threads() + 1) // 2, 8)
)
