"""Define a fake kvstore

This kvstore is used when running in the standalone mode
"""

from .. import backend as F

class KVClient(object):
    def __init__(self):
        self._data = {}
        self._push_handlers = {}
        self._pull_handlers = {}

    def barrier(self):
        pass

    def register_push_handler(self, name, func):
        self._push_handlers[name] = func

    def register_pull_handler(self, name, func):
        self._pull_handlers[name] = func

    def add_data(self, name, tensor):
        self._data[name] = tensor

    def init_data(self, name, shape, dtype, part_policy, init_func):
        self._data[name] = init_func(shape, dtype)

    def data_name_list(self):
        return list(self._data.keys())

    def get_data_meta(self, name):
        return F.dtype(self._data[name]), F.shape(self._data[name]), None

    def push(self, name, id_tensor, data_tensor):
        if name in self._push_handlers:
            self._push_handlers[name](self._data, name, id_tensor, data_tensor)
        else:
            self._data[name][id_tensor] = data_tensor

    def pull(self, name, id_tensor):
        if name in self._pull_handlers:
            return self._pull_handlers[name](self._data, name, id_tensor)
        else:
            return self._data[name][id_tensor]
