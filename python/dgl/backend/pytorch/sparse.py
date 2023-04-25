import torch as th

from ..._sparse_ops import (
    _bwd_segment_cmp,
    _csrmask,
    _csrmm,
    _csrsum,
    _edge_softmax_backward,
    _edge_softmax_forward,
    _gather_mm,
    _gather_mm_scatter,
    _gsddmm,
    _gsddmm_hetero,
    _gspmm,
    _gspmm_hetero,
    _scatter_add,
    _segment_mm,
    _segment_mm_backward_B,
    _segment_reduce,
    _update_grad_minmax_hetero,
)

from ...base import ALL, is_all
from ...heterograph_index import create_unitgraph_from_csr

__all__ = [
    "gspmm",
    "gsddmm",
    "gspmm_hetero",
    "gsddmm_hetero",
    "edge_softmax",
    "edge_softmax_hetero",
    "segment_reduce",
    "scatter_add",
    "csrmm",
    "csrsum",
    "csrmask",
    "gather_mm",
    "segment_mm",
]


def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension
    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = th.nonzero(
        th.tensor(grad_shape) - th.tensor(in_shape), as_tuple=False
    )
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    if ufeat is None or efeat is None:
        return False
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _expand(x, shape):
    return x.expand(-1, *shape)


def spmm_cache_X(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SpMM forward stage."""
    if binary_op != "copy_lhs" and req_grad_Y:
        if reduce_op == "sum":
            return True
        else:
            if binary_op == "mul":
                return True
    return False


def spmm_cache_Y(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SpMM forward stage."""
    if binary_op != "copy_rhs" and req_grad_X:
        if reduce_op == "sum":
            if binary_op in ["mul", "add"]:
                return True
        else:
            if binary_op == "mul":
                return True
    return False


def spmm_cache_argX(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argX in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ["min", "max"]:
            return True
    return False


def spmm_cache_argY(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argY in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ["min", "max"]:
            return True
    return False


class empty_context:
    """Empty context that does nothing"""

    def __init__(self, *args, **kargs):
        return

    def __enter__(self, *args, **kargs):
        return self

    def __exit__(self, *args, **kargs):
        return


# Disable CUDA autocast since we have casted args manually,
# and do it only in a nested autocast context.
def _disable_autocast_if_enabled():
    if th.is_autocast_enabled():
        return th.cuda.amp.autocast(enabled=False)
    else:
        return empty_context()


def _cast_if_autocast_enabled(*args):
    if not th.is_autocast_enabled():
        return args
    else:
        return th.cuda.amp.autocast_mode._cast(
            args, th.get_autocast_gpu_dtype()
        )


class GSpMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, op, reduce_op, X, Y):
        out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
        reduce_last = _need_reduce_last_dim(X, Y)
        X_shape = X.shape if X is not None else None
        Y_shape = Y.shape if Y is not None else None
        dtype = X.dtype if X is not None else Y.dtype
        device = X.device if X is not None else Y.device
        ctx.backward_cache = (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
        )
        req_grad_X = X.requires_grad if X is not None else False
        req_grad_Y = Y.requires_grad if Y is not None else False
        if not spmm_cache_X(op, reduce_op, req_grad_X, req_grad_Y):
            X = None
        if not spmm_cache_Y(op, reduce_op, req_grad_X, req_grad_Y):
            Y = None
        if not spmm_cache_argX(op, reduce_op, req_grad_X, req_grad_Y):
            argX = None
        if not spmm_cache_argY(op, reduce_op, req_grad_X, req_grad_Y):
            argY = None
        ctx.save_for_backward(X, Y, argX, argY)
        return out

    @staticmethod
    def backward(ctx, dZ):
        (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
        ) = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensors
        if op != "copy_rhs" and ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == "sum":
                if op == "mul":
                    dX = gspmm(g_rev, "mul", "sum", dZ, Y)
                elif op == "add":
                    dX = gspmm(g_rev, "copy_lhs", "sum", dZ, Y)
                elif op == "copy_lhs":
                    dX = gspmm(g_rev, "copy_lhs", "sum", dZ, None)
            else:  # max/min
                dX = th.zeros(
                    (X_shape[0],) + dZ.shape[1:], dtype=dtype, device=device
                )
                if op == "mul":
                    grad = _expand(Y, dZ.shape[1:]).gather(0, argY.long()) * dZ
                    dX.scatter_add_(0, argX.long(), grad)
                elif op in ["add", "copy_lhs"]:
                    dX.scatter_add_(0, argX.long(), dZ)
            dX = _reduce_grad(dX, X_shape)
        else:  # X has not gradient
            dX = None
        if op != "copy_lhs" and ctx.needs_input_grad[4]:
            if reduce_op == "sum":
                if op == "mul" and reduce_last:
                    dY = gsddmm(gidx, "dot", X, dZ)
                elif op == "mul":
                    dY = gsddmm(gidx, "mul", X, dZ)
                elif op in ["add", "copy_rhs"]:
                    dY = gsddmm(gidx, "copy_rhs", X, dZ)
            else:  # max/min
                dY = th.zeros(
                    (Y_shape[0],) + dZ.shape[1:], dtype=dtype, device=device
                )
                if op == "mul":
                    grad = _expand(X, dZ.shape[1:]).gather(0, argX.long()) * dZ
                    dY.scatter_add_(0, argY.long(), grad)
                elif op in ["add", "copy_rhs"]:
                    dY.scatter_add_(0, argY.long(), dZ)
            dY = _reduce_grad(dY, Y_shape)
        else:  # Y has no gradient
            dY = None
        return None, None, None, dX, dY


class GSpMM_hetero(th.autograd.Function):
    @staticmethod
    def forward(
        ctx, gidx, op, reduce_op, X_len, *feats
    ):  # feats = lhs_data + rhs_data
        out, (argX, argY, argX_ntype, argY_etype) = _gspmm_hetero(
            gidx, op, reduce_op, X_len, feats
        )
        X, Y = feats[:X_len], feats[X_len:]
        # TODO (Israt): check target to decide src_id/dst_id?
        src_id, dst_id = gidx.metagraph.find_edge(0)
        reduce_last = _need_reduce_last_dim(X[src_id], Y[dst_id])
        X_shape = tuple(
            [X[i].shape if X[i] is not None else None for i in range(X_len)]
        )
        Y_shape = tuple(
            [Y[i].shape if Y[i] is not None else None for i in range(len(Y))]
        )
        dtype = X[src_id].dtype if X[src_id] is not None else Y[dst_id].dtype
        device = X[src_id].device if X[src_id] is not None else Y[dst_id].device
        ctx.backward_cache = (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
            X_len,
        )
        req_grad_X = tuple(
            [
                X[i].requires_grad if X[i] is not None else False
                for i in range(X_len)
            ]
        )
        req_grad_Y = tuple(
            [
                Y[i].requires_grad if Y[i] is not None else False
                for i in range(len(Y))
            ]
        )

        # checking the first relation to decide for all the relations
        if not spmm_cache_argX(
            op, reduce_op, req_grad_X[src_id], req_grad_Y[dst_id]
        ):
            argX = tuple([None] * len(X))
        if not spmm_cache_argY(
            op, reduce_op, req_grad_X[src_id], req_grad_Y[dst_id]
        ):
            argY = tuple([None] * len(X))

        ctx.save_for_backward(*feats, *argX, *argX_ntype, *argY, *argY_etype)
        return out

    @staticmethod
    def backward(ctx, *dZ):
        (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
            X_len,
        ) = ctx.backward_cache
        num_ntypes = gidx.number_of_ntypes()
        feats = ctx.saved_tensors[: -(4 * num_ntypes)]
        argX = ctx.saved_tensors[-(4 * num_ntypes) : -(3 * num_ntypes)]
        argX_ntype = ctx.saved_tensors[-(3 * num_ntypes) : -(2 * num_ntypes)]
        argY = ctx.saved_tensors[-(2 * num_ntypes) : -num_ntypes]
        argY_etype = ctx.saved_tensors[-num_ntypes:]
        X, Y = feats[:X_len], feats[X_len:]

        if op != "copy_rhs" and any([x is not None for x in X]):
            g_rev = gidx.reverse()
            if reduce_op == "sum":
                if op == "mul":
                    dX = gspmm_hetero(
                        g_rev, "mul", "sum", len(X), *tuple(dZ + Y)
                    )
                elif op == "add":
                    dX = gspmm_hetero(
                        g_rev, "copy_lhs", "sum", len(X), *tuple(dZ + Y)
                    )
                elif op == "copy_lhs":
                    tpl_None = tuple([None] * len(Y))
                    dX = gspmm_hetero(
                        g_rev, "copy_lhs", "sum", len(X), *tuple(dZ + tpl_None)
                    )
            else:  # max/min
                # Assuming that the features are of the same dimension (enforced by the forward function)
                src_id, dst_id = gidx.metagraph.find_edge(0)
                dX = tuple(
                    [
                        th.zeros(
                            (X_shape[i][0],) + dZ[dst_id].shape[1:],
                            dtype=dtype,
                            device=device,
                        )
                        if X[i] is not None
                        else None
                        for i in range(len(X))
                    ]
                )
                if op == "mul":
                    grad = _expand(Y, dZ.shape[1:]).gather(0, argY.long()) * dZ
                    dX.scatter_add_(0, argX.long(), grad)
                elif op in ["add", "copy_lhs"]:
                    dX = _update_grad_minmax_hetero(
                        g_rev, op, dZ, argX, argX_ntype, dX
                    )
            dX = tuple(
                [
                    _reduce_grad(dX[i], X_shape[i])
                    if X[i] is not None
                    else None
                    for i in range(len(X))
                ]
            )
        else:  # X has not gradient
            dX = tuple([None] * len(X))
        if op != "copy_lhs" and any([y is not None for y in Y]):
            # TODO(Israt): implement other combinations of reduce functions
            if reduce_op == "sum":
                tpl_dZ = tuple(
                    [
                        dZ[i] if dZ[i] is not None else None
                        for i in range(len(dZ))
                    ]
                )
                tpl_X_dZ = tuple(X + tpl_dZ)
                if op == "mul" and reduce_last:
                    dY = gsddmm_hetero(gidx, "dot", X_len, "u", "v", *tpl_X_dZ)
                elif op == "mul":
                    dY = gsddmm_hetero(gidx, "mul", X_len, "u", "v", *tpl_X_dZ)
                elif op in ["add", "copy_rhs"]:
                    dY = gsddmm_hetero(
                        gidx, "copy_rhs", X_len, "u", "v", *tpl_X_dZ
                    )
            else:  # max/min
                src_id, dst_id = gidx.metagraph.find_edge(0)
                dY = tuple(
                    [
                        th.zeros(
                            (Y_shape[i][0],) + dZ[dst_id].shape[1:],
                            dtype=dtype,
                            device=device,
                        )
                        if Y[i] is not None
                        else None
                        for i in range(len(Y))
                    ]
                )
                if op == "mul":
                    grad = _expand(X, dZ.shape[1:]).gather(0, argX.long()) * dZ
                    dY.scatter_add_(0, argY.long(), grad)
                elif op in ["add", "copy_rhs"]:
                    dY = _update_grad_minmax_hetero(
                        gidx.reverse(), op, dZ, argY, argY_etype, dY
                    )
            dY = tuple(
                [
                    _reduce_grad(dY[i], Y_shape[i])
                    if dY[i] is not None
                    else None
                    for i in range(len(dY))
                ]
            )
        else:  # Y has no gradient
            dY = tuple([None] * len(Y))
        return (None, None, None, None) + dX + dY


def sddmm_cache_X(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SDDMM forward stage."""
    if op in ["mul", "dot"] and req_grad_Y:
        return True
    return False


def sddmm_cache_Y(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SDDMM forward stage."""
    if op in ["mul", "dot"] and req_grad_X:
        return True
    return False


class GSDDMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, op, X, Y, lhs_target, rhs_target):
        out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
        X_shape = X.shape if X is not None else None
        Y_shape = Y.shape if Y is not None else None
        ctx.backward_cache = gidx, op, lhs_target, rhs_target, X_shape, Y_shape
        req_grad_X = X.requires_grad if X is not None else False
        req_grad_Y = Y.requires_grad if Y is not None else False
        if not sddmm_cache_X(op, req_grad_X, req_grad_Y):
            X = None
        if not sddmm_cache_Y(op, req_grad_X, req_grad_Y):
            Y = None
        ctx.save_for_backward(X, Y)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, lhs_target, rhs_target, X_shape, Y_shape = ctx.backward_cache
        X, Y = ctx.saved_tensors
        if op != "copy_rhs" and ctx.needs_input_grad[2]:
            if lhs_target in ["u", "v"]:
                _gidx = gidx if lhs_target == "v" else gidx.reverse()
                if op in ["add", "copy_lhs"]:
                    dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ)
                else:  # mul, dot
                    if rhs_target == lhs_target:
                        dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ) * Y
                    elif rhs_target == "e":
                        dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ * Y)
                    else:  # rhs_target = !lhs_target
                        dX = gspmm(_gidx, "mul", "sum", Y, dZ)
            else:  # lhs_target == 'e'
                if op in ["add", "copy_lhs"]:
                    dX = dZ
                else:  # mul, dot
                    dX = gsddmm(gidx, "mul", dZ, Y, "e", rhs_target)
            dX = _reduce_grad(dX, X_shape)
        else:
            dX = None
        if op != "copy_lhs" and ctx.needs_input_grad[3]:
            if rhs_target in ["u", "v"]:
                _gidx = gidx if rhs_target == "v" else gidx.reverse()
                if op in ["add", "copy_rhs"]:
                    dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ)
                else:  # mul, dot
                    if lhs_target == rhs_target:
                        dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ) * X
                    elif lhs_target == "e":
                        dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ * X)
                    else:  # rhs_target = !lhs_target
                        dY = gspmm(_gidx, "mul", "sum", X, dZ)
            else:
                if op in ["add", "copy_rhs"]:
                    dY = dZ
                else:  # mul, dot
                    dY = gsddmm(gidx, "mul", dZ, X, "e", lhs_target)
            dY = _reduce_grad(dY, Y_shape)
        else:
            dY = None
        return None, None, dX, dY, None, None


class GSDDMM_hetero(th.autograd.Function):
    @staticmethod
    def forward(
        ctx, gidx, op, X_len, lhs_target, rhs_target, *feats
    ):  # feats = X+Y
        out = _gsddmm_hetero(gidx, op, X_len, lhs_target, rhs_target, feats)
        X, Y = feats[:X_len], feats[X_len:]
        X_shape = tuple(
            [X[i].shape if X[i] is not None else None for i in range(len(X))]
        )
        Y_shape = tuple(
            [Y[i].shape if Y[i] is not None else None for i in range(len(Y))]
        )
        ctx.backward_cache = (
            gidx,
            op,
            lhs_target,
            rhs_target,
            X_shape,
            Y_shape,
            X_len,
        )
        req_grad_X = tuple(
            [
                X[i].requires_grad if X[i] is not None else False
                for i in range(len(X))
            ]
        )
        req_grad_Y = tuple(
            [
                Y[i].requires_grad if Y[i] is not None else False
                for i in range(len(Y))
            ]
        )
        ctx.save_for_backward(*feats)
        return out

    @staticmethod
    # TODO(Israt): Implement the complete backward operator
    def backward(ctx, *dZ):
        (
            gidx,
            op,
            lhs_target,
            rhs_target,
            X_shape,
            Y_shape,
            X_len,
        ) = ctx.backward_cache
        feats = ctx.saved_tensors
        X, Y = feats[:X_len], feats[X_len:]
        if op != "copy_rhs" and any([x is not None for x in X]):
            if lhs_target in ["u", "v"]:
                _gidx = gidx if lhs_target == "v" else gidx.reverse()
                tpl_of_None = tuple([None] * len(X))
                if op in ["add", "copy_lhs"]:
                    dX = gspmm_hetero(
                        _gidx,
                        "copy_rhs",
                        "sum",
                        len(X),
                        *(tuple(tpl_of_None + dZ))
                    )
                else:  # mul, dot
                    if rhs_target == lhs_target:
                        dX = (
                            gspmm_hetero(
                                _gidx,
                                "copy_rhs",
                                "sum",
                                len(X),
                                *(tuple(tpl_of_None + dZ))
                            )
                            * Y
                        )
                    elif rhs_target == "e":
                        dZ_mul_Y = tuple(
                            [
                                dZ[i] * Y[i] if dZ[i] is not None else None
                                for i in range(len(Y))
                            ]
                        )
                        dX = gspmm_hetero(
                            _gidx,
                            "copy_rhs",
                            "sum",
                            len(X),
                            *(tuple(tpl_of_None + dZ_mul_Y))
                        )
                    else:  # rhs_target = !lhs_target
                        dX = gspmm_hetero(
                            _gidx, "mul", "sum", len(X), *tuple(Y + dZ)
                        )
            else:  # lhs_target == 'e'
                if op in ["add", "copy_lhs"]:
                    dX = dZ
                else:  # mul, dot
                    num_etype = gidx.number_of_etypes()
                    dX = gsddmm_hetero(
                        gidx, "mul", num_etype, "e", rhs_target, *tuple(dZ + Y)
                    )
            dX = tuple(
                [
                    _reduce_grad(dX[i], X_shape[i])
                    if X[i] is not None
                    else None
                    for i in range(len(X))
                ]
            )
        else:
            dX = tuple([None] * len(X))
        if op != "copy_lhs" and any([y is not None for y in Y]):
            if rhs_target in ["u", "v"]:
                _gidx = gidx if rhs_target == "v" else gidx.reverse()
                tpl_of_None = tuple([None] * len(X))
                if op in ["add", "copy_rhs"]:
                    dY = gspmm_hetero(
                        _gidx,
                        "copy_rhs",
                        "sum",
                        len(X),
                        *(tuple(tpl_of_None + dZ))
                    )
                else:  # mul, dot
                    if lhs_target == rhs_target:
                        dY = (
                            gspmm_hetero(
                                _gidx,
                                "copy_rhs",
                                "sum",
                                len(X),
                                *(tuple(tpl_of_None + dZ))
                            )
                            * X
                        )
                    elif lhs_target == "e":
                        dZ_mul_X = tuple(
                            [
                                dZ[i] * X[i] if dZ[i] is not None else None
                                for i in range(len(X))
                            ]
                        )
                        dY = gspmm_hetero(
                            _gidx,
                            "copy_rhs",
                            "sum",
                            len(X),
                            *(tuple(tpl_of_None + dZ_mul_X))
                        )
                    else:  # rhs_target = !lhs_target
                        dY = gspmm_hetero(
                            _gidx, "mul", "sum", len(X), *tuple(X + dZ)
                        )
            else:
                if op in ["add", "copy_rhs"]:
                    dY = tuple(
                        [
                            dZ[i] if dZ[i] is not None else None
                            for i in range(len(dZ))
                        ]
                    )
                else:  # mul, dot
                    num_etype = gidx.number_of_etypes()
                    dY = gsddmm_hetero(
                        gidx, "mul", num_etype, "e", lhs_target, *tuple(dZ + X)
                    )
            dY = tuple(
                [
                    _reduce_grad(dY[i], Y_shape[i])
                    if Y[i] is not None
                    else None
                    for i in range(len(Y))
                ]
            )
        else:
            dY = tuple([None] * len(Y))
        return (None, None, None, None, None) + dX + dY


class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, score, eids, norm_by):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        # remember to save the graph to backward cache before making it
        # a local variable
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()
        # Note: Now _edge_softmax_forward op only supports CPU
        # TODO(Zhejiang): We will support GPU in the future
        if score.is_cuda:
            score_max = _gspmm(gidx, "copy_rhs", "max", None, score)[0]
            score = th.exp(_gsddmm(gidx, "sub", score, score_max, "e", "v"))
            score_sum = _gspmm(gidx, "copy_rhs", "sum", None, score)[0]
            out = _gsddmm(gidx, "div", score, score_sum, "e", "v")
        else:
            out = _edge_softmax_forward(gidx, score, "copy_rhs")
        ctx.backward_cache = gidx
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - out * sds_sum  # multiple expressions
            return grad_score.data
        """
        gidx = ctx.backward_cache
        (out,) = ctx.saved_tensors
        sds = out * grad_out
        # Note: Now _edge_softmax_backward op only supports CPU
        # TODO(Zhejiang): We will support GPU in the future
        if out.is_cuda:
            accum = gspmm(gidx, "copy_rhs", "sum", None, sds)

            grad_score = sds - gsddmm(gidx, "mul", out, accum, "e", "v")
        else:
            grad_score = _edge_softmax_backward(gidx, out, sds)
        return None, grad_score, None, None


class EdgeSoftmax_hetero(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, eids, norm_by, *score):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        # remember to save the graph to backward cache before making it
        # a local variable
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()
        u_len = gidx.number_of_ntypes()
        e_len = gidx.number_of_etypes()
        lhs = [None] * u_len
        feats = tuple(lhs + list(score))
        score_max = _gspmm_hetero(gidx, "copy_rhs", "max", u_len, feats)[0]
        out_tmp = _gsddmm_hetero(
            gidx, "sub", e_len, "e", "v", tuple(list(score) + list(score_max))
        )
        score = tuple(
            [
                th.exp(out_tmp[i]) if out_tmp[i] is not None else None
                for i in range(len(out_tmp))
            ]
        )
        score_sum = _gspmm_hetero(
            gidx, "copy_rhs", "sum", u_len, tuple(lhs + list(score))
        )[0]
        out = _gsddmm_hetero(
            gidx, "div", e_len, "e", "v", tuple(list(score) + list(score_sum))
        )
        ctx.backward_cache = gidx
        ctx.save_for_backward(*out)
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - out * sds_sum  # multiple expressions
            return grad_score.data
        """
        gidx = ctx.backward_cache
        u_len = gidx.number_of_ntypes()
        e_len = gidx.number_of_etypes()
        lhs = [None] * u_len
        out = ctx.saved_tensors
        sds = tuple([out[i] * grad_out[i] for i in range(len(out))])
        accum = _gspmm_hetero(
            gidx, "copy_rhs", "sum", u_len, tuple(lhs + list(sds))
        )[0]
        out_sddmm = _gsddmm_hetero(
            gidx, "mul", e_len, "e", "v", tuple(list(out) + list(accum))
        )
        grad_score = tuple([sds[i] - out_sddmm[i] for i in range(len(sds))])
        return (None, None, None) + grad_score


class SegmentReduce(th.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, offsets):
        y, arg = _segment_reduce(op, x, offsets)
        ctx.save_for_backward(arg, offsets)
        ctx.backward_cache = op
        return y

    @staticmethod
    def backward(ctx, dy):
        op = ctx.backward_cache
        arg, offsets = ctx.saved_tensors
        m = offsets[-1].item()
        if op == "sum":
            offsets = offsets[1:]
            # To address the issue of trailing zeros, related issue:
            # https://github.com/dmlc/dgl/pull/2610
            indices = th.zeros(
                (m + 1,), device=offsets.device, dtype=offsets.dtype
            )
            indices.scatter_add_(0, offsets, th.ones_like(offsets))
            indices = th.cumsum(indices, -1)[:-1]
            dx = dy[indices]
        else:
            dx = _bwd_segment_cmp(dy, arg, m)
        return None, dx, None


class ScatterAdd(th.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, m):
        y = _scatter_add(x, idx, m)
        ctx.save_for_backward(idx)
        return y

    @staticmethod
    def backward(ctx, dy):
        idx = ctx.saved_tensors
        return dy[idx], None, None


class CSRMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidxA, A_weights, gidxB, B_weights, num_vtypes):
        gidxC, C_weights = _csrmm(
            gidxA, A_weights, gidxB, B_weights, num_vtypes
        )
        (
            nrows,
            ncols,
            C_indptr,
            C_indices,
            C_eids,
        ) = gidxC.adjacency_matrix_tensors(0, False, "csr")
        # Note: the returned C_indptr, C_indices and C_eids tensors MUST be the same
        # as the underlying tensors of the created graph gidxC.
        ctx.backward_cache = gidxA, gidxB, gidxC
        ctx.save_for_backward(A_weights, B_weights)
        return (
            th.tensor(nrows),
            th.tensor(ncols),
            C_indptr,
            C_indices,
            C_eids,
            C_weights,
        )

    @staticmethod
    def backward(
        ctx, dnrows, dncols, dC_indptr, dC_indices, dC_eids, dC_weights
    ):
        # Only the last argument is meaningful.
        gidxA, gidxB, gidxC = ctx.backward_cache
        A_weights, B_weights = ctx.saved_tensors
        dgidxA, dA_weights = csrmm(
            gidxC,
            dC_weights,
            gidxB.reverse(),
            B_weights,
            gidxA.number_of_ntypes(),
        )
        dgidxB, dB_weights = csrmm(
            gidxA.reverse(),
            A_weights,
            gidxC,
            dC_weights,
            gidxB.number_of_ntypes(),
        )
        dA_weights = csrmask(dgidxA, dA_weights, gidxA)
        dB_weights = csrmask(dgidxB, dB_weights, gidxB)
        return None, dA_weights, None, dB_weights, None


class CSRSum(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidxs, *weights):
        # PyTorch tensors must be explicit arguments of the forward function
        gidxC, C_weights = _csrsum(gidxs, weights)
        (
            nrows,
            ncols,
            C_indptr,
            C_indices,
            C_eids,
        ) = gidxC.adjacency_matrix_tensors(0, False, "csr")
        # Note: the returned C_indptr, C_indices and C_eids tensors MUST be the same
        # as the underlying tensors of the created graph gidxC.
        ctx.backward_cache = gidxs, gidxC
        return (
            th.tensor(nrows),
            th.tensor(ncols),
            C_indptr,
            C_indices,
            C_eids,
            C_weights,
        )

    @staticmethod
    def backward(
        ctx, dnrows, dncols, dC_indptr, dC_indices, dC_eids, dC_weights
    ):
        # Only the last argument is meaningful.
        gidxs, gidxC = ctx.backward_cache
        return (None,) + tuple(
            csrmask(gidxC, dC_weights, gidx) for gidx in gidxs
        )


class CSRMask(th.autograd.Function):
    @staticmethod
    def forward(ctx, gidxA, A_weights, gidxB):
        ctx.backward_cache = gidxA, gidxB
        return _csrmask(gidxA, A_weights, gidxB)

    @staticmethod
    def backward(ctx, dB_weights):
        gidxA, gidxB = ctx.backward_cache
        return None, csrmask(gidxB, dB_weights, gidxA), None


class SEGMENTMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, seglen_A):
        if B.dim() != 3:
            raise ValueError("segment_mm expects B to be a 3D tensor.")
        C = th.empty((A.shape[0], B.shape[2]), device=A.device, dtype=A.dtype)
        C = _segment_mm(A, B, C, seglen_A)
        ctx.backward_cache = A, B, seglen_A
        return C

    @staticmethod
    def backward(ctx, dZ):
        A, B, seglen_A = ctx.backward_cache
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            #  Compute A_grad = Out_grad * B^T
            A_grad = th.empty(A.shape, device=A.device, dtype=A.dtype)
            A_grad = _segment_mm(dZ, B, A_grad, seglen_A, b_trans=True)
        if ctx.needs_input_grad[1]:
            #  Compute B_grad = A^T * Out_grad
            B_grad = th.empty(B.shape, device=B.device, dtype=B.dtype)
            B_grad = _segment_mm_backward_B(A, dZ, B_grad, seglen_A)
        return A_grad, B_grad, None


class GATHERMM(th.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, idx_a, idx_b):
        if B.dim() != 3:
            raise ValueError(
                "Expected dimension of B is 3. Got " + str(B.dim())
            )
        N = len(idx_b) if idx_a is None else len(idx_a)
        C = th.zeros((N, B.shape[2]), device=A.device, dtype=A.dtype)
        C = _gather_mm(A, B, C, idx_a, idx_b)
        ctx.backward_cache = A, B, idx_a, idx_b
        return C

    @staticmethod
    def backward(ctx, dZ):
        A, B, idx_a, idx_b = ctx.backward_cache
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            #  Compute A_grad = Out_grad * B^T
            A_grad = th.zeros(A.shape, device=A.device, dtype=A.dtype)
            A_grad = _gather_mm_scatter(
                dZ, B.transpose(1, 2), A_grad, idx_b=idx_b, idx_c=idx_a
            )
        if ctx.needs_input_grad[1]:
            #  Compute B_grad = A^T * Out_grad
            B_grad = th.zeros(B.shape, device=B.device, dtype=B.dtype)
            B_grad = _gather_mm_scatter(A, dZ, B_grad, idx_a=idx_a, idx_c=idx_b)
        return A_grad, B_grad, None, None


def gspmm(gidx, op, reduce_op, lhs_data, rhs_data):
    if op == "sub":
        op = "add"
        rhs_data = -rhs_data
    if op == "div":
        op = "mul"
        rhs_data = 1.0 / rhs_data
    args = _cast_if_autocast_enabled(gidx, op, reduce_op, lhs_data, rhs_data)
    with _disable_autocast_if_enabled():
        return GSpMM.apply(*args)


def gsddmm(gidx, op, lhs_data, rhs_data, lhs_target="u", rhs_target="v"):
    if op == "sub":
        op = "add"
        rhs_data = -rhs_data
    if op == "div":
        op = "mul"
        rhs_data = 1.0 / rhs_data
    args = _cast_if_autocast_enabled(
        gidx, op, lhs_data, rhs_data, lhs_target, rhs_target
    )
    with _disable_autocast_if_enabled():
        return GSDDMM.apply(*args)


def gspmm_hetero(g, op, reduce_op, lhs_len, *lhs_and_rhs_tuple):
    lhs_tuple, rhs_tuple = (
        lhs_and_rhs_tuple[:lhs_len],
        lhs_and_rhs_tuple[lhs_len:],
    )
    if op == "sub":
        op = "add"
        rhs_tuple = tuple(
            [
                -rhs_tuple[i] if rhs_tuple[i] is not None else None
                for i in range(len(rhs_tuple))
            ]
        )
    if op == "div":
        op = "mul"
        rhs_tuple = tuple(
            [
                (1.0 / rhs_tuple[i]) if rhs_tuple[i] is not None else None
                for i in range(len(rhs_tuple))
            ]
        )
    if op in ["add", "mul"]:
        lhs_and_rhs_tuple = tuple(list(lhs_tuple) + list(rhs_tuple))

    args = _cast_if_autocast_enabled(
        g, op, reduce_op, lhs_len, *lhs_and_rhs_tuple
    )
    with _disable_autocast_if_enabled():
        return GSpMM_hetero.apply(*args)


def gsddmm_hetero(
    g, op, lhs_len, lhs_target="u", rhs_target="v", *lhs_and_rhs_tuple
):
    lhs_tuple, rhs_tuple = (
        lhs_and_rhs_tuple[:lhs_len],
        lhs_and_rhs_tuple[lhs_len:],
    )
    if op == "sub":
        op = "add"
        rhs_tuple = tuple(
            [
                -rhs_tuple[i] if rhs_tuple[i] is not None else None
                for i in range(len(rhs_tuple))
            ]
        )
    if op == "div":
        op = "mul"
        rhs_tuple = tuple(
            [
                (1.0 / rhs_tuple[i]) if rhs_tuple[i] is not None else None
                for i in range(len(rhs_tuple))
            ]
        )
    if op in ["add", "mul"]:
        lhs_and_rhs_tuple = tuple(list(lhs_tuple) + list(rhs_tuple))

    args = _cast_if_autocast_enabled(
        g, op, lhs_len, lhs_target, rhs_target, *lhs_and_rhs_tuple
    )
    with _disable_autocast_if_enabled():
        return GSDDMM_hetero.apply(*args)


def edge_softmax(gidx, logits, eids=ALL, norm_by="dst"):
    args = _cast_if_autocast_enabled(gidx, logits, eids, norm_by)
    with _disable_autocast_if_enabled():
        return EdgeSoftmax.apply(*args)


def edge_softmax_hetero(gidx, eids=ALL, norm_by="dst", *logits):
    args = _cast_if_autocast_enabled(gidx, eids, norm_by, *logits)
    with _disable_autocast_if_enabled():
        return EdgeSoftmax_hetero.apply(*args)


def segment_reduce(op, x, offsets):
    args = _cast_if_autocast_enabled(op, x, offsets)
    with _disable_autocast_if_enabled():
        return SegmentReduce.apply(*args)


def scatter_add(x, idx, m):
    args = _cast_if_autocast_enabled(x, idx, m)
    with _disable_autocast_if_enabled():
        return ScatterAdd.apply(*args)


def csrmm(gidxA, A_weights, gidxB, B_weights, num_vtypes):
    nrows, ncols, C_indptr, C_indices, C_eids, C_weights = CSRMM.apply(
        gidxA, A_weights, gidxB, B_weights, num_vtypes
    )
    gidxC = create_unitgraph_from_csr(
        num_vtypes,
        nrows.item(),
        ncols.item(),
        C_indptr,
        C_indices,
        C_eids,
        ["coo", "csr", "csc"],
    )
    return gidxC, C_weights


def csrsum(gidxs, weights):
    nrows, ncols, C_indptr, C_indices, C_eids, C_weights = CSRSum.apply(
        gidxs, *weights
    )
    gidxC = create_unitgraph_from_csr(
        gidxs[0].number_of_ntypes(),
        nrows.item(),
        ncols.item(),
        C_indptr,
        C_indices,
        C_eids,
        ["coo", "csr", "csc"],
    )
    return gidxC, C_weights


def csrmask(gidxA, A_weights, gidxB):
    return CSRMask.apply(gidxA, A_weights, gidxB)


def segment_mm(A, B, seglen_A):
    if A.device.type == "cpu":
        C = []
        off = 0
        for i in range(B.shape[0]):
            C.append(A[off : off + seglen_A[i]] @ B[i])
            off += seglen_A[i]
        return th.cat(C)
    else:
        args = _cast_if_autocast_enabled(A, B, seglen_A)
        with _disable_autocast_if_enabled():
            return SEGMENTMM.apply(*args)


def gather_mm(A, B, idx_A=None, idx_B=None):
    if A.device.type == "cpu":
        A = A[idx_A] if idx_A is not None else A
        B = B[idx_B] if idx_B is not None else B
        return th.bmm(A.unsqueeze(1), B).squeeze(1)
    else:
        args = _cast_if_autocast_enabled(A, B, idx_A, idx_B)
        with _disable_autocast_if_enabled():
            return GATHERMM.apply(*args)
