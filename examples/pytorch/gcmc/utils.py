import csv
import re
from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim


class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, "w")
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow(
            [
                parse_format % kwargs[attr_name]
                for attr_name, parse_format in self._attr_format_dict.items()
            ]
        )
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = (
        "Total Param Number: {}\n".format(torch_total_param_num(net))
        + "Params:\n"
    )
    for k, v in net.named_parameters():
        info_str += "\t{}: {}, {}\n".format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == "leaky":
            return nn.LeakyReLU(0.1)
        elif act == "relu":
            return nn.ReLU()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "softsign":
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == "sgd":
        return optim.SGD
    elif opt == "adam":
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace(".", "_")
