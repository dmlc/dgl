"""Define a fake kvstore

This kvstore is used when running in the standalone mode
"""

from .. import backend as F


class KVClient(object):
    """The fake KVStore client.

    This is to mimic the distributed KVStore client. It's used for DistGraph
    in standalone mode.
    """

    def __init__(self):
        self._data = {}
        self._all_possible_part_policy = {}
        self._push_handlers = {}
        self._pull_handlers = {}
        # Store all graph data name
        self._gdata_name_list = set()

    @property
    def all_possible_part_policy(self):
        """Get all possible partition policies"""
        return self._all_possible_part_policy

    @property
    def num_servers(self):
        """Get the number of servers"""
        return 1

    def barrier(self):
        """barrier"""

    def register_push_handler(self, name, func):
        """register push handler"""
        self._push_handlers[name] = func

    def register_pull_handler(self, name, func):
        """register pull handler"""
        self._pull_handlers[name] = func

    def add_data(self, name, tensor, part_policy):
        """add data to the client"""
        self._data[name] = tensor
        self._gdata_name_list.add(name)
        if part_policy.policy_str not in self._all_possible_part_policy:
            self._all_possible_part_policy[part_policy.policy_str] = part_policy

    def init_data(
        self, name, shape, dtype, part_policy, init_func, is_gdata=True
    ):
        """add new data to the client"""
        self._data[name] = init_func(shape, dtype)
        if part_policy.policy_str not in self._all_possible_part_policy:
            self._all_possible_part_policy[part_policy.policy_str] = part_policy
        if is_gdata:
            self._gdata_name_list.add(name)

    def delete_data(self, name):
        """delete the data"""
        del self._data[name]
        if name in self._gdata_name_list:
            self._gdata_name_list.remove(name)

    def data_name_list(self):
        """get the names of all data"""
        return list(self._data.keys())

    def gdata_name_list(self):
        """get the names of graph data"""
        return list(self._gdata_name_list)

    def get_data_meta(self, name):
        """get the metadata of data"""
        return F.dtype(self._data[name]), F.shape(self._data[name]), None

    def push(self, name, id_tensor, data_tensor):
        """push data to kvstore"""
        if name in self._push_handlers:
            self._push_handlers[name](self._data, name, id_tensor, data_tensor)
        else:
            F.scatter_row_inplace(self._data[name], id_tensor, data_tensor)

    def pull(self, name, id_tensor):
        """pull data from kvstore"""
        if name in self._pull_handlers:
            return self._pull_handlers[name](self._data, name, id_tensor)
        else:
            return F.gather_row(self._data[name], id_tensor)

    def map_shared_data(self, partition_book):
        """Mapping shared-memory tensor from server to client."""

    def count_nonzero(self, name):
        """Count nonzero value by pull request from KVServers.

        Parameters
        ----------
        name : str
            data name

        Returns
        -------
        int
            the number of nonzero in this data.
        """
        return F.count_nonzero(self._data[name])

    @property
    def data_store(self):
        """Return the local partition of the data storage.

        Returns
        -------
        dict[str, Tensor]
            The tensor storages of the local partition.
        """
        return self._data

    def union(self, operand1_name, operand2_name, output_name):
        """Compute the union of two mask arrays in the KVStore."""
        self._data[output_name][:] = (
            self._data[operand1_name] | self._data[operand2_name]
        )
