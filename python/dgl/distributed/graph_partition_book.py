"""Define graph partition book."""

import json
import numpy as np

from .. import backend as F
from ..base import NID, EID
from ..data.utils import load_graphs

class GraphPartitionBook:
    """Partition information.

    Note that, we assume that all partitions exists in local machines for now.
    Once we have the reshuffle() api we can change that.
    """
    def __init__(self, partition_config_file, ip_config_file):
        """Initialization

        Parameters
        ----------
        partition_config_file : str
            path of graph partition file.
        ip_config_file : str
            path of IP configuration file.
        """
        with open(partition_config_file) as conf_f:
            self._part_meta = json.load(conf_f)
        # Read ip list from ip_config_file
        self._ip_list = []
        lines = [line.rstrip('\n') for line in open(ip_config_file)]
        # we assume that ip is sorted by machine ID in ip_config_file
        for line in lines:
            ip_addr, _, _ = line.split(' ')
            self._ip_list.append(ip_addr)
        # Get number of partitions
        assert 'num_parts' in self._part_meta, "cannot get the number of partitions."
        self._num_partitions = self._part_meta['num_parts']
        assert self._num_partitions > 0, 'num_partitions cannot be a negative number.'
        # Get part_files
        self._part_files = []
        for part_id in range(self._num_partitions):
            assert 'part-{}'.format(part_id) in self._part_meta, \
            "part-{} does not exist".format(part_id)
            part_files = self._part_meta['part-{}'.format(part_id)]
            self._part_files.append(part_files)
        # Get nid2partid
        assert 'node_map' in self._part_meta, "cannot get the node map."
        self._nid2partid = F.tensor(np.load(self._part_meta['node_map']))
        # Get eid2partid
        assert 'edge_map' in self._part_meta, "cannot get the edge map."
        self._eid2partid = F.tensor(np.load(self._part_meta['edge_map']))
        # Get meta data
        self._meta_data = []
        _, nid_count = np.unique(F.asnumpy(self._nid2partid), return_counts=True)
        _, eid_count = np.unique(F.asnumpy(self._eid2partid), return_counts=True)
        for part_id in range(self._num_partitions):
            part_info = {}
            part_info['machine_id'] = part_id
            part_info['ip'] = self._ip_list[part_id]
            part_info['num_nodes'] = nid_count[part_id]
            part_info['num_edges'] = eid_count[part_id]
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
        # Get part_graphs
        self._part_graphs = []
        for part_id in range(self._num_partitions):
            graph = load_graphs(self._part_files[part_id]['part_graph'])[0][0]
            self._part_graphs.append(graph)
        # Get nidg2l
        self._nidg2l = []
        for partid in range(self._num_partitions):
            global_id = self._part_graphs[partid].ndata[NID]
            max_global_id = np.amax(F.asnumpy(global_id))
            g2l = F.zeros((max_global_id+1), F.int64, F.cpu())
            g2l[global_id] = F.arange(0, len(global_id))
            self._nidg2l.append(g2l)
        # Get eidg2l
        self._eidg2l = []
        for part_id in range(self._num_partitions):
            global_id = self._part_graphs[partid].edata[EID]
            max_global_id = np.amax(F.asnumpy(global_id))
            g2l = F.zeros((max_global_id+1), F.int64, F.cpu())
            g2l[global_id] = F.arange(0, len(global_id))
            self._eidg2l.append(g2l)


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
        return self._part_graphs[partid]
