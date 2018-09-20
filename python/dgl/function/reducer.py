"""Built-in reducer function."""
from __future__ import absolute_import

import dgl.backend as F

__all__ = ["ReduceFunction", "sum", "max"]

class ReduceFunction(object):
    def __call__(self, node, msgs):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def is_spmv_supported(self):
        raise NotImplementedError

class BundledReduceFunction(ReduceFunction):
    def __init__(self, fn_list):
        if not isinstance(fn_list, (list, tuple)):
            fn_list = [fn_list]
        else:
            # sanity check on out field
            for fn in fn_list:
                if isinstance(fn, ReduceFunction) and fn.out_field is None:
                    raise RuntimeError("Not specifying out field for multiple reduce is ambiguous")
        self.fn_list = fn_list

    def is_spmv_supported(self):
        for fn in self.fn_list:
            if not isinstance(fn, ReduceFunction) or not fn.is_spmv_supported():
                return False
        return True

    def __call__(self, node, msgs):
        ret = None
        for fn in self.fn_list:
            rpr = fn(node, msgs)
            if ret is None:
                ret = rpr
            else:
                try:
                    # ret and rpr must be dict
                    ret.update(rpr)
                except:
                    raise RuntimeError("Must specify out field for multiple reudce")
        return ret

    def name(self):
        return "bundled"

class ReducerFunctionTemplate(ReduceFunction):
    def __init__(self, name, batch_op, nonbatch_op, msg_field=None, out_field=None):
        self.name = name
        self.batch_op = batch_op
        self.nonbatch_op = nonbatch_op
        self.msg_field = msg_field
        self.out_field = out_field

    def is_spmv_supported(self):
        # TODO: support max
        return self.name == "sum"

    def __call__(self, node, msgs):
        if isinstance(msgs, list):
            if self.msg_field is None:
                ret = self.nonbatch_op(msgs)
            else:
                ret = self.nonbatch_op([msg[self.msg_field] for msg in msgs])
        else:
            if self.msg_field is None:
                ret = self.batch_op(msgs, 1)
            else:
                ret = self.batch_op(msgs[self.msg_field], 1)
        if self.out_field is None:
            return ret
        else:
            return {self.out_field : ret}

    def name(self):
        return self.name

_python_sum = sum
def sum(msgs=None, out=None):
    return ReducerFunctionTemplate("sum", F.sum, _python_sum, msgs, out)

_python_max = max
def max(msgs=None, out=None):
    return ReducerFunctionTemplate("max", F.max, _python_max, msgs, out)
