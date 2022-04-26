import jax
import jax.numpy as jnp

@jax.custom_vjp  # no nondiff_argnums!
def clip_gradient(lo, hi, x):
  return x  # identity function

def clip_gradient_fwd(lo, hi, x):
  return x, (lo, hi)  # save lo and hi values as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (None, None, jnp.clip(g, lo, hi))  # return None for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

grad_clip_gradient = jax.grad(clip_gradient)
print(grad_clip_gradient(0.0, 1.0, 0.5))
