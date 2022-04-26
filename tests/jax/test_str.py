import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def op(x, y, op):
    if op == "add": z = x + y
    if op == "mul": z = x * y
    return z

def op_fwd(x, y, op):
    cache = (x, y)
    if op == "add": z = x + y
    if op == "mul": z = x * y
    return z, cache

def op_bwd(op, cache, dz):
    x, y = cache
    if op == "add": dz_dx = dz_dy = dz
    if op == "mul": dz_dx, dz_dy = dz * y, dz * x
    return (dz_dx, dz_dy)

op.defvjp(op_fwd, op_bwd)
grad_op = jax.grad(op)
grad_op(1.0, 1.0, "add")
