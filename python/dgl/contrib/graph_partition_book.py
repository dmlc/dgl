import os
import sys
import numpy as np

from .. import backend as F
from ..base import NID, EID
from ..data.utils import load_graphs, load_tensors

class GraphPartitionBook:
    def __init__(self, part_metadata, ip_config_file):
        """Partition information.

        Parameters
        ----------
        part_metadata : json object
            metadata of partitioned graph, which is created by load_partition() API.
        ip_config_file : str
            path of IP configuration file.
        """
        self._part_meta = part_metadata
        self._meta_data = [] # list[dict[str, any]]
        self._nid2partid = None
        self._eid2partid = None
        self._partid2nids = []
        self._partid2eids = []
        self._nidg2l = []
        self._eidg2l = []
        # Get number of partitions
        assert 'num_parts' in self._part_meta, "cannot get the number of partitions."
        self._num_partitions = self._part_meta['num_parts']
        assert self._num_partitions > 0, 'num_partitions cannot be a negative number.'
        # Get part_files
        self._part_files = []
        for part_id in range(self._num_partitions):
            assert 'part-{}'.format(part_id) in self._part_meta, "part-{} does not exist".format(part_id)
            part_files = self._part_meta['part-{}'.format(part_id)]
            self._part_files.append(part_files)
        # Get part_graphs
        self._part_graphs = []
        for part_id in range(self._num_partitions):
            graph = load_graphs(self._part_files[part_id]['part_graph'])[0][0]
            self._part_graphs.append(graph)
        # Read ip list from ip_config_file
        self._ip_list = []
        lines = [line.rstrip('\n') for line in open(ip_config_file)]
        for line in lines:
            ip, _, _ = line.split(' ')
            self._ip_list.append(ip)


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
        if len(self._meta_data) == 0:
            for part_id in range(self._num_partitions):
                part_info = {}
                part_info['machine_id'] = part_id
                part_info['ip'] = self._ip_list[part_id]
                node_feats = load_tensors(self._part_files[part_id]['node_feats'])
                edge_feats = load_tensors(self._part_files[part_id]['edge_feats'])
                part_info['num_nodes'] = len(node_feats)
                part_info['num_edges'] = len(edge_feats)
                self._meta_data.append(part_info)

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
        if self._nid2partid is None:
            assert 'node_map' in self._part_meta, "cannot get the node map."
            self._nid2partid = np.load(self._part_meta['node_map'])

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
        if self._eid2partid is None:
            assert 'edge_map' in self._part_meta, "cannot get the edge map."
            self._eid2partid = np.load(self._part_meta['edge_map'])

        return self._eid2partid
    

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
        if len(self._partid2nids) == 0:
            sorted_nid = F.tensor(np.argsort(F.asnumpy(self._nid2partid)))
            part, count = np.unique(F.asnumpy(self._nid2partid), return_counts=True)
            assert len(part) == self._num_partitions
            start = 0
            for offset in count:
                part_nids = sorted_nid[start:start+offset]
                start += offset
                self._partid2nids.append(part_nids)

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
        if len(self._partid2eids) == 0:
            sorted_eid = F.tensor(np.argsort(F.asnumpy(self._eid2partid)))
            part, count = np.unique(F.asnumpy(self._eid2partid), return_counts=True)
            assert len(part) == self._num_partitions
            start = 0
            for offset in count:
                part_eids = sorted_eid[start:start+offset]
                start += offset
                self._partid2eids.append(part_eids)

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
        if len(self._nidg2l) == 0:
            global_id = self._part_graphs[partid].ndata[NID]
            max_global_id = F.asnumpy(global_id).amax()
            g2l = F.zeros((max_id+1), F.int64, F.cpu())
            g2l[global_id] = F.arange(0, len(global_id))
            self._nidg2l.append(g2l)

        return self._nidg2l[partid]
        

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
        if len(self._eidg2l) == 0:
            global_id = self._part_graphs[partid].edata[EID]
            max_global_id = F.asnumpy(global_id).amax()
            g2l = F.zeros((max_id+1), F.int64, F.cpu())
            g2l[global_id] = F.arange(0, len(global_id))
            self._eidg2l.append(g2l)

        return self._eidg2l[partid]


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