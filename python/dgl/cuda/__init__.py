""" CUDA wrappers """
from . import nccl
from .._ffi.streams import stream, set_stream
