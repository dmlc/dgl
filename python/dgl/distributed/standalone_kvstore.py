"""Define a fake kvstore

This kvstore is used when running in the standalone mode
"""

from .. import backend as F

class KVClient(object):
    ''' The fake KVStore client.

    This is to mimic the distributed KVStore client. It's used for DistGraph
    in standalone mode.
    '''
    def __init__(self):
        self._data = {}
        self._push_handlers = {}
        self._pull_handlers = {}

    def barrier(self):
        '''barrier'''

    def register_push_handler(self, name, func):
        '''register push handler'''
        self._push_handlers[name] = func

    def register_pull_handler(self, name, func):
        '''register pull handler'''
        self._pull_handlers[name] = func

    def add_data(self, name, tensor):
        '''add data to the client'''
        self._data[name] = tensor

    def init_data(self, name, shape, dtype, _, init_func):
        '''add new data to the client'''
        self._data[name] = init_func(shape, dtype)

    def delete_data(self, name):
        '''delete the data'''
        del self._data[name]

    def data_name_list(self):
        '''get the names of all data'''
        return list(self._data.keys())

    def get_data_meta(self, name):
        '''get the metadata of data'''
        return F.dtype(self._data[name]), F.shape(self._data[name]), None

    def push(self, name, id_tensor, data_tensor):
        '''push data to kvstore'''
        if name in self._push_handlers:
            self._push_handlers[name](self._data, name, id_tensor, data_tensor)
        else:
            F.scatter_row_inplace(self._data[name], id_tensor, data_tensor)

    def pull(self, name, id_tensor):
        '''pull data from kvstore'''
        if name in self._pull_handlers:
            return self._pull_handlers[name](self._data, name, id_tensor)
        else:
            return F.gather_row(self._data[name], id_tensor)
