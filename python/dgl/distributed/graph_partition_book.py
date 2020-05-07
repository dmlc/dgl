"""Define graph partition book."""

import numpy as np

from .. import backend as F
from ..base import NID, EID

class GraphPartitionBook:
    """GraphPartitionBook is used to store parition information.

    Parameters
    ----------
    part_id : int
        partition id of current GraphPartitionBook
    partition_meta : tuple
        partition meta data created by partition_graph() API, including
        (num_nodes, num_edges, node_map, edge_map, num_parts)
    local_graph : DGLGraph
        The graph partition structure.
    ip_config_file : str
        path of IP configuration file. The format of configuration file should be:

          [ip] [base_port] [server_count]

          172.31.40.143 30050 2
          172.31.36.140 30050 2
          172.31.47.147 30050 2
          172.31.30.180 30050 2

         we assume that ip is sorted by machine ID in ip_config_file.
    """
    def __init__(self, part_id, partition_meta, local_graph, ip_config_file):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert len(partition_meta) == 5, \
        'partition_meta must include: (num_nodes, num_edges, node_map, edge_map, num_parts)'
        self._part_id = part_id
        self._graph = local_graph
        _, _, self._nid2partid, self._eid2partid, self._num_partitions = partition_meta
        self._nid2partid = F.tensor(self._nid2partid)
        self._eid2partid = F.tensor(self._eid2partid)
        # Read ip list from ip_config_file
        self._ip_list = []
        lines = [line.rstrip('\n') for line in open(ip_config_file)]
        for line in lines:
            ip_addr, _, _ = line.split(' ')
            self._ip_list.append(ip_addr)
        # Get meta data of GraphPartitionBook
        self._meta_data = []
        _, nid_count = np.unique(F.asnumpy(self._nid2partid), return_counts=True)
        _, eid_count = np.unique(F.asnumpy(self._eid2partid), return_counts=True)
        for partid in range(self._num_partitions):
            part_info = {}
            part_info['machine_id'] = partid
            part_info['ip'] = self._ip_list[partid]
            part_info['num_nodes'] = nid_count[partid]
            part_info['num_edges'] = eid_count[partid]
            self._meta_data.append(part_info)
        # Get partid2nids
        self._partid2nids = []
        sorted_nid = F.tensor(np.argsort(F.asnumpy(self._nid2partid)))
        start = 0
        for offset in nid_count:
            part_nids = sorted_nid[start:start+offset]
            start += offset
            self._partid2nids.append(part_nids)
        # Get partid2eids
        self._partid2eids = []
        sorted_eid = F.tensor(np.argsort(F.asnumpy(self._eid2partid)))
        start = 0
        for offset in eid_count:
            part_eids = sorted_eid[start:start+offset]
            start += offset
            self._partid2eids.append(part_eids)
        # Get nidg2l
        self._nidg2l = [None] * self._num_partitions
        global_id = self._graph.ndata[NID]
        max_global_id = np.amax(F.asnumpy(global_id))
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l[global_id] = F.arange(0, len(global_id))
        self._nidg2l[self._part_id] = g2l
        # Get eidg2l
        self._eidg2l = [None] * self._num_partitions
        global_id = self._graph.edata[EID]
        max_global_id = np.amax(F.asnumpy(global_id))
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l[global_id] = F.arange(0, len(global_id))
        self._eidg2l[self._part_id] = g2l


    def num_partitions(self):
        """Return the number of partitions.

        Returns
        -------
        int
            number of partitions
        """
        return self._num_partitions


    def metadata(self):
        """Return the partition meta data.

        The meta data includes:

        * The machine ID.
        * The machine IP address.
        * Number of nodes and edges of each partition.

        Examples
        --------
        >>> print(g.get_partition_book().metadata())
        >>> [{'machine_id' : 0, 'ip': '192.168.8.12', 'num_nodes' : 3000, 'num_edges' : 5000},
        ...  {'machine_id' : 1, 'ip': '192.168.8.13', 'num_nodes' : 2000, 'num_edges' : 4888},
        ...  ...]

        Returns
        -------
        list[dict[str, any]]
            Meta data of each partition.
        """
        return self._meta_data


    def nid2partid(self, nids):
        """From global node IDs to partition IDs

        Parameters
        ----------
        nids : tensor
            global node IDs

        Returns
        -------
        tensor
            partition IDs
        """
        return self._nid2partid[nids]


    def eid2partid(self, eids):
        """From global edge IDs to partition IDs

        Parameters
        ----------
        eids : tensor
            global edge IDs

        Returns
        -------
        tensor
            partition IDs
        """
        return self._eid2partid[eids]


    def partid2nids(self, partid):
        """From partition id to node IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            node IDs
        """
        return self._partid2nids[partid]


    def partid2eids(self, partid):
        """From partition id to edge IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            edge IDs
        """
        return self._partid2eids[partid]


    def nid2localnid(self, nids, partid):
        """Get local node IDs within the given partition.

        Parameters
        ----------
        nids : tensor
            global node IDs
        partid : int
            partition ID

        Returns
        -------
        tensor
             local node IDs
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote tensor of nid2localnid.')

        return self._nidg2l[partid][nids]


    def eid2localeid(self, eids, partid):
        """Get the local edge ids within the given partition.

        Parameters
        ----------
        eids : tensor
            global edge ids
        partid : int
            partition ID

        Returns
        -------
        tensor
             local edge ids
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote tensor of eid2localeid.')

        return self._eidg2l[partid][eids]


    def get_partition(self, partid):
        """Get the graph of one partition.

        Parameters
        ----------
        partid : int
            Partition ID.

        Returns
        -------
        DGLGraph
            The graph of the partition.
        """
        if partid != self._part_id:
            raise RuntimeError('Now GraphPartitionBook does not support \
                getting remote partitions.')

        return self._graph
