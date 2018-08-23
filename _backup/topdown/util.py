
import torch as T
import torch.nn.functional as F
import numpy as NP
import os

USE_CUDA = os.getenv('USE_CUDA', None)

def cuda(x, device=None, async_=False):
    return x.cuda() if USE_CUDA else x

def tovar(x, *args, dtype='float32', **kwargs):
    if not T.is_tensor(x):
        x = T.from_numpy(NP.array(x, dtype=dtype))
    return T.autograd.Variable(cuda(x), *args, **kwargs)

def tonumpy(x):
    if isinstance(x, T.autograd.Variable):
        x = x.data
    if T.is_tensor(x):
        x = x.cpu().numpy()
    return x

def create_onehot(idx, size):
    onehot = tovar(T.zeros(*size))
    onehot = onehot.scatter(1, idx.unsqueeze(1), 1)
    return onehot

def reverse(x, dim):
    idx = T.arange(x.size()[dim] - 1, -1, -1).long().to(x.device)
    return x.index_select(dim, idx)

def addbox(ax, b, ec, lw=1):
    import matplotlib.patches as PA
    ax.add_patch(PA.Rectangle((b[0] - b[2] / 2, b[1] - b[3] / 2), b[2], b[3],
                 ec=ec, fill=False, lw=lw))

def overlay(fore, fore_bbox, back):
    batch_size = fore.size()[0]
    crows, ccols = fore.size()[-2:]
    cx, cy, w, h = T.unbind(fore_bbox, -1)
    x1 = -2 * cx / w
    x2 = 2 * (1 - cx) / w
    y1 = -2 * cy / h
    y2 = 2 * (1 - cy) / h

    x1 = x1[:, None]
    x2 = x2[:, None]
    y1 = y1[:, None]
    y2 = y2[:, None]

    nrows, ncols = back.size()[-2:]
    grid_x = x1 + (x2 - x1) * tovar(T.arange(ncols))[None, :] / (ncols - 1)
    grid_y = y1 + (y2 - y1) * tovar(T.arange(nrows))[None, :] / (nrows - 1)
    grid = T.stack([
        grid_x[:, None, :].expand(batch_size, nrows, ncols),
        grid_y[:, :, None].expand(batch_size, nrows, ncols),
        ], -1)

    fore = T.cat([fore, tovar(T.ones(batch_size, 1, crows, ccols))], 1)

    fore = F.grid_sample(fore, grid)
    fore_rgb = fore[:, :3]
    fore_alpha = fore[:, 3:4]
    result = fore_rgb * fore_alpha + back * (1 - fore_alpha)

    return result

def intersection(a, b):
    x1 = T.max(a[..., 0] - a[..., 2] / 2, b[..., 0] - b[..., 2] / 2)
    y1 = T.max(a[..., 1] - a[..., 3] / 2, b[..., 1] - b[..., 3] / 2)
    x2 = T.min(a[..., 0] + a[..., 2] / 2, b[..., 0] + b[..., 2] / 2)
    y2 = T.min(a[..., 1] + a[..., 3] / 2, b[..., 1] + b[..., 3] / 2)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return w * h

def iou(a, b):
    i_area = intersection(a, b)
    a_area = a[..., 2] * a[..., 3]
    b_area = b[..., 2] * b[..., 3]
    return i_area / (a_area + b_area - i_area)
