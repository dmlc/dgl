import jax
import jax.numpy as jnp


# @jax.jit
# def scatter_add(x, index, source):
#     pointers = jnp.meshgrid(*[jnp.arange(axis) for axis in x.shape], indexing="ij")
#     pointers = jnp.stack(pointers, axis=-1)
#     pointers = pointers.reshape(-1, pointers.shape[-1])
#     pointers_x = jnp.stack([index[tuple(pointer)] for pointer in pointers])
#     pointers_x = jnp.expand_dims(pointers_x, -1)
#     pointers_x = jnp.concatenate(
#         [
#             pointers_x,
#             pointers[:, 1:]
#         ],
#         axis=-1
#     )
#
#     for pointer_x, pointer in zip(pointers_x, pointers):
#         x = x.at[pointer_x].add(source[pointer])
#
#     return x

@jax.jit
def scatter_add(x, index, source):
    pointers = jnp.meshgrid(*[jnp.arange(axis) for axis in x.shape], sparse=True, indexing="ij")
    pointers[0] = index
    x = x.at[tuple(pointers)].add(source)
    # print(x[tuple(pointers)].shape)

    return x


def run():
    from jax import random
    import time
    key = random.PRNGKey(2666)
    x = jnp.zeros((8, 8, 8, 8))
    source = jnp.ones((8, 8, 8, 8))
    index = jax.random.categorical(key, logits=jnp.zeros(8), shape=(8, 8, 8, 8))
    time0 = time.time()
    scatter_add(x, index, source)
    time1 = time.time()
    print(time1 - time0)

if __name__ == "__main__":
    run()
