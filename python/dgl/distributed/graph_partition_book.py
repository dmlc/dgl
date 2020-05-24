"""Define graph partition book."""

import numpy as np

from .. import backend as F
from ..base import NID, EID
from .. import utils

class GraphPartitionBook:
    """GraphPartitionBook is used to store parition information.

    Parameters
    ----------
    part_id : int
        partition id of current GraphPartitionBook
    num_parts : int
        number of total partitions
    node_map : tensor
        global node id mapping to partition id
    edge_map : tensor
        global edge id mapping to partition id
    part_graph : DGLGraph
        The graph partition structure.
    """
    def __init__(self, part_id, num_parts, node_map, edge_map, part_graph):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert num_parts > 0, 'num_parts must be greater than zero.'
        self._part_id = part_id
        self._num_partitions = num_parts
        node_map = utils.toindex(node_map)
        self._nid2partid = node_map.tousertensor()
        edge_map = utils.toindex(edge_map)
        self._eid2partid = edge_map.tousertensor()
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
        global_id = part_graph.ndata[NID]
        max_global_id = np.amax(F.asnumpy(global_id))
        # TODO(chao): support int32 index
        g2l = F.zeros((max_global_id+1), F.int64, F.context(global_id))
        g2l = F.scatter_row(g2l, global_id, F.arange(0, len(global_id)))
        self._nidg2l[self._part_id] = g2l
        # Get eidg2l
        self._eidg2l = [None] * self._num_partitions
        global_id = part_graph.edata[EID]
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
        """From partition id to global node IDs

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
        """From partition id to global edge IDs

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


class RangePartitionBook:
    """GraphPartitionBook is used to store parition information.

    Parameters
    ----------
    part_id : int
        partition id of current GraphPartitionBook
    num_parts : int
        number of total partitions
    node_map : tensor
        global node id mapping to partition id
    edge_map : tensor
        global edge id mapping to partition id
    """
    def __init__(self, part_id, num_parts, node_map, edge_map):
        assert part_id >= 0, 'part_id cannot be a negative number.'
        assert num_parts > 0, 'num_parts must be greater than zero.'
        self._part_id = part_id
        self._num_partitions = num_parts
        self._node_map = node_map
        self._edge_map = edge_map
        # Get meta data of GraphPartitionBook
        self._partition_meta_data = []
        for partid in range(self._num_partitions):
            nrange_start = node_map[partid - 1] if partid > 0 else 0
            nrange_end = node_map[partid]
            erange_start = edge_map[partid - 1] if partid > 0 else 0
            erange_end = edge_map[partid]
            part_info = {}
            part_info['machine_id'] = partid
            part_info['num_nodes'] = nrange_end - nrange_start
            part_info['num_edges'] = erange_end - erange_start
            self._partition_meta_data.append(part_info)


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
        ret = _CAPI_DGLRangeSearch(self._node_map, nids)
        ret = utils.toindex(ret)
        return ret.tousertensor()


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
        ret = _CAPI_DGLRangeSearch(self._edge_map, eids)
        ret = utils.toindex(ret)
        return ret.tousertensor()


    def partid2nids(self, partid):
        """From partition id to global node IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            node IDs
        """
        # TODO do we need to cache it?
        start = self._node_map[partid - 1] if partid > 0 else 0
        end = self._node_map[partid]
        return F.arange(start, end)


    def partid2eids(self, partid):
        """From partition id to global edge IDs

        Parameters
        ----------
        partid : int
            partition id

        Returns
        -------
        tensor
            edge IDs
        """
        # TODO do we need to cache it?
        start = self._edge_map[partid - 1] if partid > 0 else 0
        end = self._edge_map[partid]
        return F.arange(start, end)


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

        start = self._node_map[partid - 1] if partid > 0 else 0
        end = self._node_map[partid]
        assert F.sum((nids >= start) * (nids < end), 0) == len(nids)
        return nids - start


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

        start = self._edge_map[partid - 1] if partid > 0 else 0
        end = self._edge_map[partid]
        assert F.sum((eids >= start) * (eids < end), 0) == len(eids)
        return eids - start


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
