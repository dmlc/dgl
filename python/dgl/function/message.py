"""Built-in message function."""
from __future__ import absolute_import

import operator
import dgl.backend as F

__all__ = ["MessageFunction", "src_mul_edge", "copy_src", "copy_edge"]


class MessageFunction(object):
    def __call__(self, src, edge):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def is_spmv_supported(self, g):
        raise NotImplementedError


class BundledMessageFunction(MessageFunction):
    def __init__(self, fn_list):
        if not isinstance(fn_list, (list, tuple)):
            fn_list = [fn_list]
        else:
            # sanity check on out field
            for fn in fn_list:
                # cannot perform check for udf
                if isinstance(fn, MessageFunction) and fn.out_field is None:
                    raise RuntimeError("Not specifying out field for multiple message is ambiguous")
        self.fn_list = fn_list

    def is_spmv_supported(self, g):
        for fn in self.fn_list:
            if not isinstance(fn, MessageFunction) or not fn.is_spmv_supported(g):
                return False
        return True

    def __call__(self, src, edge):
        ret = None
        for fn in self.fn_list:
            msg = fn(src, edge)
            if ret is None:
                ret = msg
            else:
                try:
                    # ret and msg must be dict
                    ret.update(msg)
                except:
                    raise RuntimeError("Must specify out field for multiple message")
        return ret

    def name(self):
        return "bundled"


def _is_spmv_supported_node_feat(g, field):
    if field is None:
        feat = g.get_n_repr()
    else:
        feat = g.get_n_repr()[field]
    shape = F.shape(feat)
    return len(shape) == 1 or len(shape) == 2

def _is_spmv_supported_edge_feat(g, field):
    # check shape, only scalar edge feature can be optimized at the moment
    if field is None:
        feat = g.get_e_repr()
    else:
        feat = g.get_e_repr()[field]
    shape = F.shape(feat)
    return len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)


class SrcMulEdgeMessageFunction(MessageFunction):
    def __init__(self, mul_op, src_field=None, edge_field=None, out_field=None):
        self.mul_op = mul_op
        self.src_field = src_field
        self.edge_field = edge_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        return _is_spmv_supported_node_feat(g, self.src_field) \
                and _is_spmv_supported_edge_feat(g, self.edge_field)

    def __call__(self, src, edge):
        if self.src_field is not None:
            src = src[self.src_field]
        if self.edge_field is not None:
            edge = edge[self.edge_field]
        ret = self.mul_op(src, edge)
        if self.out_field is None:
            return ret
        else:
            return {self.out_field : ret}

    def name(self):
        return "src_mul_edge"

class CopySrcMessageFunction(MessageFunction):
    def __init__(self, src_field=None, out_field=None):
        self.src_field = src_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        return _is_spmv_supported_node_feat(g, self.src_field)

    def __call__(self, src, edge):
        if self.src_field is not None:
            ret = src[self.src_field]
        else:
            ret = src
        if self.out_field is None:
            return ret
        else:
            return {self.out_field : ret}

    def name(self):
        return "copy_src"

class CopyEdgeMessageFunction(MessageFunction):
    def __init__(self, edge_field=None, out_field=None):
        self.edge_field = edge_field
        self.out_field = out_field

    def is_spmv_supported(self, g):
        # TODO: support this with g-spmv
        return False
        # return _is_spmv_supported_edge_feat(g, self.edge_field)

    def __call__(self, src, edge):
        if self.edge_field is not None:
            ret = edge[self.edge_field]
        else:
            ret = edge
        if self.out_field is None:
            return ret
        else:
            return {self.out_field : ret}

    def name(self):
        return "copy_edge"


def src_mul_edge(src=None, edge=None, out=None):
    """TODO(minjie): docstring """
    return SrcMulEdgeMessageFunction(operator.mul, src, edge, out)

def copy_src(src=None, out=None):
    """TODO(minjie): docstring """
    return CopySrcMessageFunction(src, out)

def copy_edge(edge=None, out=None):
    """TODO(minjie): docstring """
    return CopyEdgeMessageFunction(edge, out)
