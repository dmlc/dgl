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
    num_parts : int
        number of total partitions
    node_map : numpy array
        global node id mapping to partition id
    edge_map : numpy array
        global edge id mapping to partition id
    part_graph : DGLGraph
        The graph partition structure.
    """
    def __init__(self, part_id, num_parts, node_map, edge_map, part_graph):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert num_parts > 0, 'num_parts must be greater than zero.'
        self._part_id = part_id
        self._num_partitions = num_parts
        self._nid2partid = F.zerocopy_from_numpy(node_map)
        self._eid2partid = F.zerocopy_from_numpy(edge_map)
        self._graph = part_graph
        # Get meta data of GraphPartitionBook
        self._partition_meta_data = []
        _, nid_count = np.unique(F.asnumpy(self._nid2partid), return_counts=True)
        _, eid_count = np.unique(F.asnumpy(self._eid2partid), return_counts=True)
        for partid in range(self._num_partitions):
            part_info = {}
            part_info['machine_id'] = partid
            part_info['num_nodes'] = nid_count[partid]
            part_info['num_edges'] = eid_count[partid]
            self._partition_meta_data.append(part_info)
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
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
        self._nidg2l[self._part_id] = g2l
        # Get eidg2l
        self._eidg2l = [None] * self._num_partitions
        global_id = self._graph.edata[EID]
        max_global_id = np.amax(F.asnumpy(global_id))
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
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
        >>> [{'machine_id' : 0, 'num_nodes' : 3000, 'num_edges' : 5000},
        ...  {'machine_id' : 1, 'num_nodes' : 2000, 'num_edges' : 4888},
        ...  ...]

        Returns
        -------
        list[dict[str, any]]
            Meta data of each partition.
        """
        return self._partition_meta_data


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
        return F.gather_row(self._nid2partid, nids)


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
        return F.gather_row(self._eid2partid, eids)


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

        return F.gather_row(self._nidg2l[partid], nids)


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

        return F.gather_row(self._eidg2l[partid], eids)


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
