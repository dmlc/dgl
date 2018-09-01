"""Built-in message function."""
from __future__ import absolute_import

import operator

__all__ = ["MessageFunction", "src_mul_edge", "copy_src", "copy_edge"]

class MessageFunction(object):
    def __call__(self, src, edge):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class BundledMessageFunction(MessageFunction):
    def __init__(self, fn_list):
        self.fn_list = fn_list

    def __call__(self, src, edge):
        ret = None
        for fn in self.fn_list:
            msg = fn(src, edge)
            if ret is None:
                ret = msg
            else:
                try:
                    ret.update(msg)
                except e:
                    raise RuntimeError("Failed to merge results of two builtin"
                                       " message functions. Please specify out_field"
                                       " for the builtin message function.")
        return ret

    def name(self):
        return "bundled"

class SrcMulEdgeMessageFunction(MessageFunction):
    def __init__(self, mul_op, src_field=None, edge_field=None, out_field=None):
        self.mul_op = mul_op
        self.src_field = src_field
        self.edge_field = edge_field
        self.out_field = out_field

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
