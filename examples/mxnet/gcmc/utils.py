import ast
import os
import csv
import inspect
import logging
import re
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from collections import OrderedDict

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()

def parse_ctx(ctx_args):
    ctx = re.findall('([a-z]+)(\d*)', ctx_args)
    ctx = [(device, int(num)) if len(num) > 0 else (device, 0) for device, num in ctx]
    ctx = [mx.Context(*ele) for ele in ctx]
    return ctx


def gluon_total_param_num(net):
    return sum([np.prod(v.shape) for v in net.collect_params().values()])


def gluon_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(gluon_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.collect_params().items():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def params_clip_global_norm(param_dict, clip, ctx):
    grads = [p.grad(ctx) for p in param_dict.values()]
    gnorm = gluon.utils.clip_global_norm(grads, clip)
    return gnorm

def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or HybridBlock

    Returns
    -------
    ret: HybridBlock
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
            return nn.Activation(act)
        else:
            raise NotImplementedError
    else:
        return act
