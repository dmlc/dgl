from .heterograph import DGLHeteroGraph
from .function.message import CopyMessageFunction
from .function import src_mul_edge
from contextlib import contextmanager

ORIGINAL_UPDATEALL = DGLHeteroGraph.update_all

def edata_update_fixture(edata_name_fixture):
    def new_update_all(self, message_func,
                       reduce_func,
                       apply_node_func=None,
                       etype=None):
        if isinstance(message_func, CopyMessageFunction):
            message_func = src_mul_edge(
                message_func.in_field, edata_name_fixture, message_func.out_field)
        return ORIGINAL_UPDATEALL(self, message_func, reduce_func, apply_node_func, etype)
    return new_update_all

@contextmanager
def use_edata_for_update(edata_name, verbose=False):
    DGLHeteroGraph.update_all = edata_update_fixture(edata_name)
    try:
        yield None
    finally:
        DGLHeteroGraph.update_all = ORIGINAL_UPDATEALL
