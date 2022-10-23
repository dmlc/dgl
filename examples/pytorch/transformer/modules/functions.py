import torch as th


def src_dot_dst(src_field, dst_field, out_field):
    """
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function.
    """

    def func(edges):
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def scaled_exp(field, c):
    """
    This function applies $exp(x / c)$ for input $x$, which is required by *Scaled Dot-Product Attention* mentioned in the paper.
    """

    def func(edges):
        return {field: th.exp((edges.data[field] / c).clamp(-10, 10))}

    return func
