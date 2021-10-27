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
                dgl_warning("Using edata {} for update_all computation".format(
                    edata_name_fixture))
        return ORIGINAL_UPDATEALL(self, message_func, reduce_func, apply_node_func, etype)
    return new_update_all


@contextmanager
def use_edata_for_update(edata_name, verbose=False):
    """
    Use given edata as weight when aggregating information from source nodes. 
    For example is the code inside the block originally called
    `g.udpate_all(fn.copy_u('feat', 'h'), fn.sum('h', 'out'))` will be automatically
    converted to `g.udpate_all(fn.u_mul_e('feat', edata_name, 'h'), fn.sum('h', 'out'))`.

    Only `fn.copy_u` function will be patched for update. UDF and other message functions
    are not supported.

    You can use this function with any DGL NN modules. 

    Example:
    >>> g.edata['e'] = th.randn(g.num_edges(), requires_grad=True)
    >>> with use_edata_for_update('e'):
    >>>     # This will use g.edata['e'] in the real update computation
    >>>     g.update_all(fn.copy_u('feat', 'h'), fn.sum('h', 'out'))
    >>> loss = g.ndata['out'].sum()
    >>> loss.backward()
    >>> print(g.edata['e'].grad) # g.edata['e'] will have gradient

    Parameters
    ----------
    edata_name : str
        The name of edata to use in the update_all call
    verbose: Optional[bool]
        Whether to print the logs for the patch operation. Mainly for debug purpose
    """
    DGLHeteroGraph.update_all = _edata_update_fixture(edata_name, verbose)
    try:
        yield None
    finally:
        DGLHeteroGraph.update_all = ORIGINAL_UPDATEALL
