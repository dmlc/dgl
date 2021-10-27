from .heterograph import DGLHeteroGraph
from .function.message import CopyMessageFunction, TargetCode
from .function import src_mul_edge
from .base import dgl_warning
from contextlib import contextmanager


ORIGINAL_UPDATEALL = DGLHeteroGraph.update_all

def _edata_update_fixture(edata_name_fixture, verbose):
    def new_update_all(self, message_func,
                       reduce_func,
                       apply_node_func=None,
                       etype=None):
        if isinstance(message_func, CopyMessageFunction) and message_func.target == TargetCode.SRC:
            message_func = src_mul_edge(
                message_func.in_field, edata_name_fixture, message_func.out_field)
            if verbose:
                dgl_warning("Using edata {} for update_all computation".format(edata_name_fixture))
        return ORIGINAL_UPDATEALL(self, message_func, reduce_func, apply_node_func, etype)
    return new_update_all

@contextmanager
def use_edata_for_update(edata_name, verbose=False):
    """
    Use given edata for update_all functions

    This function will change the copy_u function to u_mul_e with the edata name provided

    Parameters
    ----------
    edata_name : str
        The name of edata to use in the update_all call
    verbose: Optional[bool]
        Whether to print logs for the patched function


    """
    DGLHeteroGraph.update_all = _edata_update_fixture(edata_name, verbose)
    try:
        yield None
    finally:
        DGLHeteroGraph.update_all = ORIGINAL_UPDATEALL
