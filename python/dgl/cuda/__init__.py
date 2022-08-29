""" CUDA wrappers """
from . import nccl
from .._ffi.streams import stream, set_stream, get_current_stream
