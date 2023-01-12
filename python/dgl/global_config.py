"""Module for global configuration operators."""
from ._ffi.function import _init_api

__all__ = ["is_libxsmm_enabled", "use_libxsmm"]


def use_libxsmm(flag):
    r"""Set whether DGL uses libxsmm at runtime.

    Detailed information about libxsmm can be found here:
    https://github.com/libxsmm/libxsmm

    Parameters
    ----------
    flag : boolean
        If True, use libxsmm, otherwise not.

    See Also
    --------
    is_libxsmm_enabled
    """
    _CAPI_DGLConfigSetLibxsmm(flag)


def is_libxsmm_enabled():
    r"""Get whether the use_libxsmm flag is turned on.

    Returns
    ----------
    use_libxsmm_flag[boolean]
        True if the use_libxsmm flag is turned on.

    See Also
    ----------
    use_libxsmm
    """
    return _CAPI_DGLConfigGetLibxsmm()


_init_api("dgl.global_config")
