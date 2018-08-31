"""Built-in reducer function."""
from __future__ import absolute_import

import dgl.backend as F

__all__ = ["ReduceFunction", "sum", "max"]

class ReduceFunction(object):
    def __call__(self, node, msgs):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class BundledReduceFunction(ReduceFunction):
    def __init__(self, fn_list):
        self.fn_list = fn_list

    def __call__(self, node, msgs):
        ret = None
        for fn in self.fn_list:
            rpr = fn(node, msgs)
            if ret is None:
                ret = rpr
            else:
                try:
                    ret.update(rpr)
                except e:
                    raise RuntimeError("Failed to merge results of two builtin"
                                       " reduce functions. Please specify out_field"
                                       " for the builtin reduce function.")
        return ret

    def name(self):
        return "bundled"

class SumReducerFunction(ReduceFunction):
    def __init__(self, batch_sum_op, nonbatch_sum_op, msg_field=None, out_field=None):
        self.batch_sum_op = batch_sum_op
        self.nonbatch_sum_op = nonbatch_sum_op
        self.msg_field = msg_field
        self.out_field = out_field

    def __call__(self, node, msgs):
        if isinstance(msgs, list):
            if self.msg_field is None:
                ret = self.nonbatch_sum_op(msgs)
            else:
                ret = self.nonbatch_sum_op([msg[self.msg_field] for msg in msgs])
        else:
            if self.msg_field is None:
                ret = self.batch_sum_op(msgs, 1)
            else:
                ret = self.batch_sum_op(msgs[self.msg_field], 1)
        if self.out_field is None:
            return ret
        else:
            return {self.out_field : ret}

    def name(self):
        return "sum"

_python_sum = sum
def sum(msgs=None, out=None):
    return SumReducerFunction(F.sum, _python_sum, msgs, out)

_python_max = max
def max(msgs=None, out=None):
    return SumReducerFunction(F.max, _python_max, msgs, out)
