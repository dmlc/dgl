import jax
import jax.numpy as jnp
from functools import partial

OP_TO_IDX, IDX_TO_OP = {"add": 0, "mul": 1}, {0: "add", 1: "mul"}

# @jax.tree_util.register_pytree_node_class
# class OpStr:
#     def __init__(self, op):
#         if isinstance(op, str):
#             assert op in OP_TO_IDX, "can only be one of the ops"
#         self.op = op
#
#     def __eq__(self, other):
#         return self.op == other
#
#     def __hash__(self):
#         return hash(self.op)
#
#     def tree_flatten(self):
#         return [OP_TO_IDX[self.op]], None
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         idx = children[0]
#         if isinstance(idx, int):
#             return cls(op=IDX_TO_OP[idx])
#         else:
#             return None


@jax.custom_vjp
def op(x, y, op):
    if op == "add": z = x + y
    if op == "mul": z = x * y
    return z

def op_fwd(x, y, op):
    cache = (x, y, op)
    if op == "add": z = x + y
    if op == "mul": z = x * y
    return z, cache

def op_bwd(cache, dz):
    x, y, op = cache
    if op == "add": dz_dx = dz_dy = 1.0
    if op == "mul": dz_dx, dz_dy = y, x
    return (dz_dx, dz_dy, None)

op.defvjp(op_fwd, op_bwd)
grad_op = jax.grad(op)
grad_op(1.0, 1.0, OpStr("add"))
