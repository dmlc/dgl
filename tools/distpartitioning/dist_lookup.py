import copy
import logging
import os

import numpy as np
import pyarrow
import torch
from gloo_wrapper import allgather_sizes, alltoallv_cpu
from pyarrow import csv
from utils import map_partid_rank


class DistLookupService:
    """
    This is an implementation of a Distributed Lookup Service to provide the following
    services to its users. Map 1) global node-ids to partition-ids, and 2) global node-ids
    to shuffle global node-ids (contiguous, within each node for a give node_type and across
    all the partitions)

    This services initializes itself with the node-id to partition-id mappings, which are inputs
    to this service. The node-id to partition-id  mappings are assumed to be in one file for each
    node type. These node-id-to-partition-id mappings are split within the service processes so that
    each process ends up with a contiguous chunk. It first divides the no of mappings (node-id to
    partition-id) for each node type into equal chunks across all the service processes. So each
    service process will be thse owner of a set of node-id-to-partition-id mappings. This class
    has two functions which are as follows:

    1) `get_partition_ids` function which returns the node-id to partition-id mappings to the user
    2) `get_shuffle_nids` function which returns the node-id to shuffle-node-id mapping to the user

    Parameters:
    -----------
    input_dir : string
        string representing the input directory where the node-type partition-id
        files are located
    ntype_names : list of strings
        list of strings which are used to read files located within the input_dir
        directory and these files contents are partition-id's for the node-ids which
        are of a particular node type
    id_map : dgl.distributed.id_map instance
        this id_map is used to retrieve ntype-ids, node type ids, and type_nids, per type
        node ids, for any given global node id
    rank : integer
        integer indicating the rank of a given process
    world_size : integer
        integer indicating the total no. of processes
    num_parts : integer
        interger representing the no. of partitions
    """

    def __init__(self, input_dir, ntype_names, rank, world_size, num_parts):
        assert os.path.isdir(input_dir)
        assert ntype_names is not None
        assert len(ntype_names) > 0

        # These lists are indexed by ntype_ids.
        type_nid_begin = []
        type_nid_end = []
        partid_list = []
        ntype_count = []
        ntypes = []

        # Iterate over the node types and extract the partition id mappings.
        for ntype in ntype_names:

            filename = f"{ntype}.txt"
            logging.debug(
                f"[Rank: {rank}] Reading file: {os.path.join(input_dir, filename)}"
            )

            read_options = pyarrow.csv.ReadOptions(
                use_threads=True,
                block_size=4096,
                autogenerate_column_names=True,
            )
            parse_options = pyarrow.csv.ParseOptions(delimiter=" ")
            ntype_partids = []
            with pyarrow.csv.open_csv(
                os.path.join(input_dir, "{}.txt".format(ntype)),
                read_options=read_options,
                parse_options=parse_options,
            ) as reader:
                for next_chunk in reader:
                    if next_chunk is None:
                        break
                    next_table = pyarrow.Table.from_batches([next_chunk])
                    ntype_partids.append(next_table["f0"].to_numpy())

            ntype_partids = np.concatenate(ntype_partids)
            count = len(ntype_partids)
            ntype_count.append(count)
            ntypes.append(ntype)

            # Each rank assumes a contiguous set of partition-ids which are equally split
            # across all the processes.
            split_size = np.ceil(count / np.int64(world_size)).astype(np.int64)
            start, end = (
                np.int64(rank) * split_size,
                np.int64(rank + 1) * split_size,
            )
            if rank == (world_size - 1):
                end = count
            type_nid_begin.append(start)
            type_nid_end.append(end)

            # Slice the partition-ids which belong to the current instance.
            partid_list.append(copy.deepcopy(ntype_partids[start:end]))

            # Explicitly release the array read from the file.
            del ntype_partids

        logging.debug(
            f"[Rank: {rank}] ntypeid begin - {type_nid_begin} - {type_nid_end}"
        )

        # Store all the information in the object instance variable.
        self.type_nid_begin = np.array(type_nid_begin, dtype=np.int64)
        self.type_nid_end = np.array(type_nid_end, dtype=np.int64)
        self.partid_list = partid_list
        self.ntype_count = np.array(ntype_count, dtype=np.int64)
        self.ntypes = ntypes
        self.rank = rank
        self.world_size = world_size
        self.num_parts = num_parts

    def set_idMap(self, id_map):
        self.id_map = id_map

    def get_partition_ids(self, agg_global_nids):
        """
        This function is used to get the partition-ids for a given set of global node ids

        global_nids <-> partition-ids mappings are deterministically  distributed across
        all the participating processes, within the service. A contiguous global-nids
        (ntype-ids, per-type-nids) are stored within each process and this is determined
        by the total no. of nodes of a given ntype-id and the rank of the process.

        Process, where the global_nid <-> partition-id mapping is stored can be easily computed
        as described above. Once this is determined we perform an alltoallv to send the request.
        On the receiving side, each process receives a set of global_nids and retrieves corresponding
        partition-ids using locally stored lookup tables. It builds responses to all the other
        processes and performs alltoallv.

        Once the response, partition-ids, is received, they are re-ordered corresponding to the
        incoming global-nids order and returns to the caller.

        Parameters:
        -----------
        self : instance of this class
            instance of this class, which is passed by the runtime implicitly

        agg_global_nids : numpy array
            an array of aggregated global node-ids for which partition-ids are
            to be retrieved by the distributed lookup service.

        Returns:
        --------
        list of integers :
            list of integers, which are the partition-ids of the global-node-ids (which is the
            function argument)
        """
        CHUNK_SIZE = 200 * 1000 * 1000
        # Determine the no. of times each process has to send alltoall messages.
        local_rows = agg_global_nids.shape[0]
        all_sizes = allgather_sizes(
            [local_rows], self.world_size, self.num_parts, return_sizes=True
        )
        max_count = np.amax(all_sizes)

        if max_count <= 0:
            logging.debug(
                f"[Rank: {self.rank}] No process has global_nids to process !!!"
            )
            return

        num_splits = np.ceil(max_count / CHUNK_SIZE).astype(np.uint16)
        LOCAL_CHUNK_SIZE = np.ceil(local_rows / num_splits).astype(np.int64)
        agg_partition_ids = []

        logging.debug(
            f"[Rank: {self.rank}] BatchSize: {CHUNK_SIZE}, \
                            max_count: {max_count}, \
                            splits: {num_splits}, \
                            rows: {agg_global_nids.shape}, \
                            local batch_size: {LOCAL_CHUNK_SIZE}"
        )

        for split in range(num_splits):
            # Compute the global_nids for this iteration
            global_nids = agg_global_nids[
                split * LOCAL_CHUNK_SIZE : (split + 1) * LOCAL_CHUNK_SIZE
            ]

            # Find the process where global_nid --> partition-id(owner) is stored.
            if len(global_nids) > 0:
                ntype_ids, type_nids = self.id_map(global_nids)
                ntype_ids, type_nids = ntype_ids.numpy(), type_nids.numpy()
            else:
                ntype_ids = np.array([], dtype=np.int64)
                type_nids = np.array([], dtype=np.int64)

            assert len(ntype_ids) == len(global_nids)

            # For each node-type, the per-type-node-id <-> partition-id mappings are
            # stored as contiguous chunks by this lookup service.
            # The no. of these mappings stored by each process, in the lookup service, are
            # equally split among all the processes in the lookup service, deterministically.
            typeid_counts = self.ntype_count[ntype_ids]
            chunk_sizes = np.ceil(typeid_counts / self.world_size).astype(
                np.int64
            )
            service_owners = np.floor_divide(type_nids, chunk_sizes).astype(
                np.int64
            )

            # Now `service_owners` is a list of ranks (process-ids) which own the corresponding
            # global-nid <-> partition-id mapping.

            # Split the input global_nids into a list of lists where each list will be
            # sent to the respective rank/process
            # We also need to store the indices, in the indices_list, so that we can re-order
            # the final result (partition-ids) in the same order as the global-nids (function argument)
            send_list = []
            indices_list = []
            for idx in range(self.world_size):
                idxes = np.where(service_owners == idx)
                ll = global_nids[idxes[0]]
                send_list.append(torch.from_numpy(ll))
                indices_list.append(idxes[0])
            assert len(np.concatenate(indices_list)) == len(global_nids)
            assert np.all(
                np.sort(np.concatenate(indices_list))
                == np.arange(len(global_nids))
            )

            # Send the request to everyone else.
            # As a result of this operation, the current process also receives a list of lists
            # from all the other processes.
            # These lists are global-node-ids whose global-node-ids <-> partition-id mappings
            # are owned/stored by the current process
            owner_req_list = alltoallv_cpu(
                self.rank, self.world_size, send_list
            )

            # Create the response list here for each of the request list received in the previous
            # step. Populate the respective partition-ids in this response lists appropriately
            out_list = []
            for idx in range(self.world_size):
                if owner_req_list[idx] is None:
                    out_list.append(torch.empty((0,), dtype=torch.int64))
                    continue
                # Get the node_type_ids and per_type_nids for the incoming global_nids.
                ntype_ids, type_nids = self.id_map(owner_req_list[idx].numpy())
                ntype_ids, type_nids = ntype_ids.numpy(), type_nids.numpy()

                # Lists to store partition-ids for the incoming global-nids.
                type_id_lookups = []
                local_order_idx = []

                # Now iterate over all the node_types and acculumulate all the partition-ids
                # since all the partition-ids are based on the node_type order... they
                # must be re-ordered as per the order of the input, which may be different.
                for tid in range(len(self.partid_list)):
                    cond = ntype_ids == tid
                    local_order_idx.append(np.where(cond)[0])
                    global_type_nids = type_nids[cond]
                    if len(global_type_nids) <= 0:
                        continue

                    local_type_nids = (
                        global_type_nids - self.type_nid_begin[tid]
                    )

                    assert np.all(local_type_nids >= 0)
                    assert np.all(
                        local_type_nids
                        <= (
                            self.type_nid_end[tid]
                            + 1
                            - self.type_nid_begin[tid]
                        )
                    )

                    cur_owners = self.partid_list[tid][local_type_nids]
                    type_id_lookups.append(cur_owners)

                # Reorder the partition-ids, so that it agrees with the input order --
                # which is the order in which the incoming message is received.
                if len(type_id_lookups) <= 0:
                    out_list.append(torch.empty((0,), dtype=torch.int64))
                else:
                    # Now reorder results for each request.
                    sort_order_idx = np.argsort(np.concatenate(local_order_idx))
                    lookups = np.concatenate(type_id_lookups)[sort_order_idx]
                    out_list.append(torch.from_numpy(lookups))

            # Send the partition-ids to their respective requesting processes.
            owner_resp_list = alltoallv_cpu(
                self.rank, self.world_size, out_list
            )

            # Owner_resp_list, is a list of lists of numpy arrays where each list
            # is a list of partition-ids which the current process requested
            # Now we need to re-order so that the parition-ids correspond to the
            # global_nids which are passed into this function.

            # Order according to the requesting order.
            # Owner_resp_list is the list of owner-ids for global_nids (function argument).
            owner_ids = [x for x in owner_resp_list if x is not None]
            if len(owner_ids) > 0:
                owner_ids = torch.cat(owner_ids).numpy()
            else:
                owner_ids = np.array([], dtype=np.int64)
            assert len(owner_ids) == len(global_nids)

            global_nids_order = np.concatenate(indices_list)
            sort_order_idx = np.argsort(global_nids_order)
            owner_ids = owner_ids[sort_order_idx]
            global_nids_order = global_nids_order[sort_order_idx]
            assert np.all(np.arange(len(global_nids)) == global_nids_order)

            if len(owner_ids) > 0:
                # Store the partition-ids for the current split
                agg_partition_ids.append(owner_ids)

        # Stitch the list of partition-ids and return to the caller
        if len(agg_partition_ids) > 0:
            agg_partition_ids = np.concatenate(agg_partition_ids)
        else:
            agg_partition_ids = np.array([], dtype=np.int64)
        assert agg_global_nids.shape[0] == agg_partition_ids.shape[0]

        # Now the owner_ids (partition-ids) which corresponding to the  global_nids.
        return agg_partition_ids

    def get_shuffle_nids(
        self, global_nids, my_global_nids, my_shuffle_global_nids, world_size
    ):
        """
        This function is used to retrieve shuffle_global_nids for a given set of incoming
        global_nids. Note that global_nids are of random order and will contain duplicates

        This function first retrieves the partition-ids of the incoming global_nids.
        These partition-ids which are also the ranks of processes which own the respective
        global-nids as well as shuffle-global-nids. alltoallv is performed to send the
        global-nids to respective ranks/partition-ids where the mapping
        global-nids <-> shuffle-global-nid is located.

        On the receiving side, once the global-nids are received associated shuffle-global-nids
        are retrieved and an alltoallv is performed to send the responses to all the other
        processes.

        Once the responses, shuffle-global-nids, are received, they are re-ordered according
        to the incoming global-nids order and returns to the caller.

        Parameters:
        -----------
        self : instance of this class
            instance of this class, which is passed by the runtime implicitly
        global_nids : numpy array
            an array of global node-ids for which partition-ids are to be retrieved by
            the distributed lookup service.
        my_global_nids: numpy ndarray
            array of global_nids which are owned by the current partition/rank/process
            This process has the node <-> partition id mapping
        my_shuffle_global_nids : numpy ndarray
            array of shuffle_global_nids which are assigned by the current process/rank
        world_size : int
            total no. of processes in the MPI_WORLD

        Returns:
        --------
        list of integers:
            list of shuffle_global_nids which correspond to the incoming node-ids in the
            global_nids.
        """

        # Get the owner_ids (partition-ids or rank).
        owner_ids = self.get_partition_ids(global_nids)

        # These owner_ids, which are also partition ids of the nodes in the
        # input graph, are in the range 0 - (num_partitions - 1).
        # These ids are generated using some kind of graph partitioning method.
        # Distribuged lookup service, as used by the graph partitioning
        # pipeline, is used to store ntype-ids (also type_nids) and their
        # mapping to the associated partition-id.
        # These ids are split into `num_process` chunks and processes in the
        # dist. lookup service are assigned the owernship of these chunks.
        # The pipeline also enforeces the following constraint among the
        # pipeline input parameters: num_partitions, num_processes
        #   num_partitions is an integer multiple of num_processes
        #   which means each individual node in the cluster will be running
        #   equal number of processes.
        owner_ids = map_partid_rank(owner_ids, world_size)

        # Ask these owners to supply for the shuffle_global_nids.
        send_list = []
        id_list = []
        for idx in range(self.world_size):
            cond = owner_ids == idx
            idxes = np.where(cond)
            ll = global_nids[idxes[0]]
            send_list.append(torch.from_numpy(ll))
            id_list.append(idxes[0])

        assert len(np.concatenate(id_list)) == len(global_nids)
        cur_global_nids = alltoallv_cpu(self.rank, self.world_size, send_list)

        # At this point, current process received a list of lists each containing
        # a list of global-nids whose corresponding shuffle_global_nids are located
        # in the current process.
        shuffle_nids_list = []
        for idx in range(self.world_size):
            if cur_global_nids[idx] is None:
                shuffle_nids_list.append(torch.empty((0,), dtype=torch.int64))
                continue

            uniq_ids, inverse_idx = np.unique(
                cur_global_nids[idx], return_inverse=True
            )
            common, idx1, idx2 = np.intersect1d(
                uniq_ids,
                my_global_nids,
                assume_unique=True,
                return_indices=True,
            )
            assert len(common) == len(uniq_ids)

            req_shuffle_global_nids = my_shuffle_global_nids[idx2][inverse_idx]
            assert len(req_shuffle_global_nids) == len(cur_global_nids[idx])
            shuffle_nids_list.append(torch.from_numpy(req_shuffle_global_nids))

        # Send the shuffle-global-nids to their respective ranks.
        mapped_global_nids = alltoallv_cpu(
            self.rank, self.world_size, shuffle_nids_list
        )
        for idx in range(len(mapped_global_nids)):
            if mapped_global_nids[idx] == None:
                mapped_global_nids[idx] = torch.empty((0,), dtype=torch.int64)

        # Reorder to match global_nids (function parameter).
        global_nids_order = np.concatenate(id_list)
        shuffle_global_nids = torch.cat(mapped_global_nids).numpy()
        assert len(shuffle_global_nids) == len(global_nids)

        sorted_idx = np.argsort(global_nids_order)
        shuffle_global_nids = shuffle_global_nids[sorted_idx]
        global_nids_ordered = global_nids_order[sorted_idx]
        assert np.all(global_nids_ordered == np.arange(len(global_nids)))

        return shuffle_global_nids
