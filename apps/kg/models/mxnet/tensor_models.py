import os
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

from .score_fun import *
from .. import *

def logsigmoid(val):
    max_elem = nd.maximum(0., -val)
    z = nd.exp(-max_elem) + nd.exp(-val - max_elem)
    return -(max_elem + nd.log(z))

get_device = lambda args : mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
norm = lambda x, p: nd.sum(nd.abs(x) ** p)

get_scalar = lambda x: x.detach().asscalar()

reshape = lambda arr, x, y: arr.reshape(x, y)

cuda = lambda arr, gpu: arr.as_in_context(mx.gpu(gpu))

class ExternalEmbedding:
    def __init__(self, args, num, dim, ctx):
        self.gpu = args.gpu
        self.args = args
        self.trace = []

        self.emb = nd.empty((num, dim), dtype=np.float32, ctx=ctx)
        self.state_sum = nd.zeros((self.emb.shape[0]), dtype=np.float32, ctx=ctx)
        self.state_step = 0

    def init(self, emb_init):
        nd.random.uniform(-emb_init, emb_init,
                          shape=self.emb.shape, dtype=self.emb.dtype,
                          ctx=self.emb.context, out=self.emb)

    def share_memory(self):
        # TODO(zhengda) fix this later
        pass

    def __call__(self, idx, gpu_id=-1, trace=True):
        if self.emb.context != idx.context:
            idx = idx.as_in_context(self.emb.context)
        data = nd.take(self.emb, idx)
        if self.gpu >= 0:
            data = data.as_in_context(mx.gpu(self.gpu))
        data.attach_grad()
        if trace:
            self.trace.append((idx, data))
        return data

    def update(self):
        self.state_step += 1
        for idx, data in self.trace:
            grad = data.grad

            clr = self.args.lr
            #clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

            # the update is non-linear so indices must be unique
            grad_indices = idx
            grad_values = grad

            grad_sum = (grad_values * grad_values).mean(1)
            ctx = self.state_sum.context
            if ctx != grad_indices.context:
                grad_indices = grad_indices.as_in_context(ctx)
            if ctx != grad_sum.context:
                grad_sum = grad_sum.as_in_context(ctx)
            self.state_sum[grad_indices] += grad_sum
            std = self.state_sum[grad_indices]  # _sparse_mask
            std_values = nd.expand_dims(nd.sqrt(std) + 1e-10, 1)
            if self.gpu >= 0:
                std_values = std_values.as_in_context(mx.gpu(self.args.gpu))
            tmp = (-clr * grad_values / std_values)
            if tmp.context != ctx:
                tmp = tmp.as_in_context(ctx)
            # TODO(zhengda) the overhead is here.
            self.emb[grad_indices] = mx.nd.take(self.emb, grad_indices) + tmp
        self.trace = []

    def curr_emb(self):
        data = [data for _, data in self.trace]
        return nd.concat(*data, dim=0)

    def save(self, path, name):
        emb_fname = os.path.join(path, name+'.npy')
        np.save(emb_fname, self.emb.asnumpy())

    def load(self, path, name):
        emb_fname = os.path.join(path, name+'.npy')
        self.emb = nd.array(np.load(emb_fname))
