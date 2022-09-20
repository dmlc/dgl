"""profiling module for DGL"""
from .._ffi.function import _init_api


def start_profiling():
    """
    Enable debug level logging for DGL
    """
    _CAPI_DGLStartProfiling()


def stop_profiling():
    """
    Enable debug level logging for DGL
    """
    _CAPI_DGLStopProfiling()


def print_profiler_stats():
    """setup logger"""
    _CAPI_DGLPrintStats()


_init_api('dgl.utils.profiler', __name__)

