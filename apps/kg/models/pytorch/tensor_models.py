"""
Knowledge Graph Embedding Models.
1. TransE
2. DistMult
3. ComplEx
4. RotatE
5. pRotatE
6. TransH
7. TransR
8. TransD
9. RESCAL
"""
import os
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT

from .. import *

logsigmoid = functional.logsigmoid

def get_device(args):
    return th.device('cpu') if args.gpu < 0 else th.device('cuda:' + str(args.gpu))

norm = lambda x, p: x.norm(p=p)**p

get_scalar = lambda x: x.detach().item()

reshape = lambda arr, x, y: arr.view(x, y)

cuda = lambda arr, gpu: arr.cuda(gpu)

class ExternalEmbedding:
    def __init__(self, args, num, dim, device):
        self.gpu = args.gpu
        self.args = args
        self.trace = []

        self.emb = th.empty(num, dim, dtype=th.float32, device=device)
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()
        self.state_step = 0

    def init(self, emb_init):
        INIT.uniform_(self.emb, -emb_init, emb_init)
        INIT.zeros_(self.state_sum)

    def share_memory(self):
        self.emb.share_memory_()
        self.state_sum.share_memory_()

    def __call__(self, idx, gpu_id=-1, trace=True):
        s = self.emb[idx]
        if self.gpu >= 0:
            s = s.cuda(self.gpu)
        data = s.clone().detach().requires_grad_(True)
        if trace:
            self.trace.append((idx, data))
        return data

    def update(self):
        self.state_step += 1
        with th.no_grad():
            for idx, data in self.trace:
                grad = data.grad.data

                clr = self.args.lr
                #clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

                # the update is non-linear so indices must be unique
                grad_indices = idx
                grad_values = grad

                grad_sum = (grad_values * grad_values).mean(1)
                device = self.state_sum.device
                if device != grad_indices.device:
                    grad_indices = grad_indices.to(device)
                if device != grad_sum.device:
                    grad_sum = grad_sum.to(device)
                self.state_sum.index_add_(0, grad_indices, grad_sum)
                std = self.state_sum[grad_indices]  # _sparse_mask
                std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                if self.gpu >= 0:
                    std_values = std_values.cuda(self.args.gpu)
                tmp = (-clr * grad_values / std_values)
                if tmp.device != device:
                    tmp = tmp.to(device)
                # TODO(zhengda) the overhead is here.
                self.emb.index_add_(0, grad_indices, tmp)
        self.trace = []

    def curr_emb(self):
        data = [data for _, data in self.trace]
        return th.cat(data, 0)

    def save(self, path, name):
        file_name = os.path.join(path, name+'.npy')
        np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        file_name = os.path.join(path, name+'.npy')
        self.emb = th.Tensor(np.load(file_name))
