"""ParMETIS distributed partitioning entry script."""
import json
import os
import argparse
import logging

import numpy as np
from numpy.lib.format import open_memmap
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.dispatchers import predicate_dispatch
from utils import files
import utils

logging.basicConfig(level='INFO')

def _save_partitioned(arr, num_partitions, output_path):
    num_rows = arr.shape[0]
    num_rows_per_partition = (num_rows + num_partitions - 1) // num_partitions
    os.makedirs(output_path, exist_ok=True)
    new_offsets = []
    for i in range(0, num_partitions):
        start = i * num_rows_per_partition
        end = min(num_rows, (i + 1) * num_rows_per_partition)
        part_path = os.path.join(output_path, '%d.npy' % i)
        outbuf = open_memmap(
                part_path, mode='w+', dtype=arr.dtype, shape=(end - start, *arr.shape[1:]))
        logging.info('Saving %d-%d to %s' % (start, end, part_path))
        with logging_redirect_tqdm():
            for j in tqdm.trange(0, end - start, 1000000):
                j_end = min(end - start, j + 1000000)
                outbuf[j:j_end] = arr[j + start:j_end + start]
        new_offsets.append((os.path.abspath(part_path), start, end))
    return new_offsets

@predicate_dispatch
def split_tensor(meta, num_partitions, output_path):
    """Split an on-disk tensor to multiple partitions and save it in :attr:`output_path`.

    Returns a list of (path, start index, end index) tuples.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta.get('fmt'))

@split_tensor.register(files.is_numpy)
def _(meta, num_partitions, output_path):
    arr = files.np_load(meta['path'])
    return _save_partitioned(arr, num_partitions, output_path)

@split_tensor.register(files.is_c_blob)
def _(meta, num_partitions, output_path):
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    arr = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    return _save_partitioned(arr, num_partitions, output_path)

@split_tensor.register(files.is_partitioned_numpy)
def _(meta, num_partitions, output_path):
    path = meta['path']
    filenames = files.get_partitioned_numpy_file_list(path)
    partition_paths = [os.path.join(path, filename) for filename in filenames]
    partition_offsets = np.zeros(len(partition_paths) + 1, dtype='int64')
    for i, path in enumerate(partition_paths):
        arr = files.np_load(path)
        partition_offsets[i + 1] = partition_offsets[i] + arr.shape[0]
        shape = arr.shape[1:]
    num_rows = partition_offsets[-1]
    shape = (num_rows,) + tuple(shape)

    num_rows_per_partition = (num_rows + num_partitions - 1) // num_partitions
    os.makedirs(output_path, exist_ok=True)

    part_id = 0
    part = files.np_load(partition_paths[part_id])
    part_offset = 0
    part_remaining = part.shape[0]
    new_offsets = []
    with tqdm.tqdm(total=num_rows) as tq:
        for i in range(0, num_partitions):
            start = i * num_rows_per_partition
            end = min(num_rows, (i + 1) * num_rows_per_partition)
            part_path = os.path.join(output_path, '%d.npy' % i)
            outbuf = open_memmap(
                    part_path, mode='w+', dtype=part.dtype, shape=(end - start, *part.shape[1:]))
            logging.info('Saving tensor to partition %d in %s (%d-%d)' %
                    (i, part_path, start, end))
            new_offsets.append((os.path.abspath(part_path), start, end))
            j = 0
            j_end = end - start

            while j < j_end:
                copysize = min(j_end - j, part_remaining)
                outbuf[j:j + copysize] = part[part_offset:part_offset + copysize]
                tq.update(copysize)
                part_remaining -= copysize
                j += copysize
                start += copysize
                if part_remaining == 0 and start < num_rows:
                    part_id += 1
                    logging.info('Loading next input partition in %s' % partition_paths[part_id])
                    part = files.np_load(partition_paths[part_id])
                    part_offset = 0
                    part_remaining = part.shape[0]

    return new_offsets

@predicate_dispatch
def split_edge_index(meta, num_partitions, output_path, utype_offset, vtype_offset,
                     cetype, cetype_idx):
    """Split an on-disk edge index into multiple ParMETIS input files, adding
    a fixed offset for both source and destination nodes for homogenization.

    Returns a list of (path, start index, end index) tuples.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta.get('fmt'))

def _save_partitioned_edge_index(
        edge_index, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
    num_edges_per_partition = (edge_index.shape[1] + num_partitions - 1) // num_partitions
    result = []
    os.makedirs(os.path.join(output_path, cetype), exist_ok=True)

    with logging_redirect_tqdm():
        for partition_id in range(num_partitions):
            start_id = partition_id * num_edges_per_partition
            end_id = min(start_id + num_edges_per_partition, edge_index.shape[1])
            file_path = os.path.join(output_path, cetype, '%s-%d.txt' % (cetype, partition_id))
            logging.info('Generating partition %d in %s (%d-%d)' % (partition_id, file_path, start_id, end_id))
            buffer_ = edge_index[:, start_id:end_id]

            out_buffer = np.zeros((end_id - start_id, 4), dtype='int64')
            out_buffer[:, 0] = buffer_[0] + utype_offset
            out_buffer[:, 1] = buffer_[1] + vtype_offset
            out_buffer[:, 2] = np.arange(start_id, end_id, dtype='int64')
            out_buffer[:, 3] = cetype_idx
            utils.savetxt(file_path, out_buffer)
            result.append([os.path.abspath(file_path), start_id, end_id])
    return result

@split_edge_index.register(files.is_numpy)
def _(meta, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
    logging.info('Loading numpy file with metadata %s' % meta)
    edge_index = files.np_load(meta['path'])
    return _save_partitioned_edge_index(
            edge_index, num_partitions, output_path, utype_offset, vtype_offset, cetype,
            cetype_idx)

@split_edge_index.register(files.is_c_blob)
def _(meta, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
    logging.info('Loading C blob file with metadata %s' % meta)
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    edge_index = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    return _save_partitioned_edge_index(
            edge_index, num_partitions, output_path, utype_offset, vtype_offset, cetype,
            cetype_idx)

@split_edge_index.register(files.is_partitioned_numpy)
def _(meta, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
    logging.info('Loading with metadata %s' % meta)
    path = meta['path']
    filenames = files.get_partitioned_numpy_file_list(path)
    partition_paths = [os.path.join(path, filename) for filename in filenames]
    partition_offsets = np.zeros(len(partition_paths) + 1, dtype='int64')
    for i, path in enumerate(partition_paths):
        arr = files.np_load(path)
        partition_offsets[i + 1] = partition_offsets[i] + arr.shape[1]
    num_edges = partition_offsets[-1]
    shape = (2, num_edges)

    result = []

    num_edges_per_partition = (num_edges + num_partitions - 1) // num_partitions
    os.makedirs(output_path, exist_ok=True)
    output_part_id = 0
    output_part_path = os.path.join(output_path, '%d.txt' % output_part_id)
    output_part_file = open(output_part_path, 'wb')
    start_id = row_id = 0
    edges_remaining = num_edges_per_partition
    for i, part_path in enumerate(partition_paths):
        logging.info('Generating partition %d in %s' % (i, part_path))
        part_edge_index = files.np_load(part_path)
        with logging_redirect_tqdm():
            logging.info('Number of edges in input partition: %d' % part_edge_index.shape[1])
            with tqdm.tqdm(total=part_edge_index.shape[1]) as tq:
                j = 0
                while j < part_edge_index.shape[1]:
                    j_end = min(j + 1000000, j + edges_remaining, part_edge_index.shape[1])

                    out_buffer = np.zeros((j_end - j, 4), dtype='int64')
                    out_buffer[:, 0] = part_edge_index[0, j:j_end] + utype_offset
                    out_buffer[:, 1] = part_edge_index[1, j:j_end] + vtype_offset
                    out_buffer[:, 2] = np.arange(j, j_end, dtype='int64')
                    out_buffer[:, 3] = cetype_idx
                    utils.savetxt(output_part_file, out_buffer)
                    edges_remaining -= j_end - j
                    tq.update(j_end - j)
                    j = j_end
                    if edges_remaining == 0 and j < part_edge_index.shape[1]:
                        output_part_file.close()
                        output_part_id += 1
                        logging.info('Saved %d to %s' % (j_end, output_part_path))
                        output_part_path = os.path.join(output_path, '%d.txt' % output_part_id)
                        logging.info('Switching output partition file to %s' % output_part_path)
                        output_part_file = open(output_part_path, 'wb')
                        edges_remaining = num_edges_per_partition
    output_part_file.close()

    return result


def _prepare_parmetis_data(output_path, meta, num_partitions):
    parmetis_output = {}
    for i, (ty, type_data) in enumerate(meta.items()):
        paths = {}
        for key, key_data in type_data.items():
            key_path = os.path.join(output_path, ty, key)
            os.makedirs(key_path, exist_ok=True)
            paths[key] = split_tensor(key_data, num_partitions, key_path)
        parmetis_output[ty] = paths
    return parmetis_output

def prepare_input(metadata, output_path, num_partitions, config):
    """Prepare the input for ParMETIS with multiple files format.
    """
    num_nodes = metadata['num_nodes']
    edges = metadata.get('edges', None)
    ndata = metadata.get('node_data', None)
    edata = metadata.get('edge_data', None)

    parmetis_nid = {}
    parmetis_eid = {}

    logging.info('Making output directory in %s' % output_path)
    os.makedirs(output_path, exist_ok=True)
    cwd = os.getcwd()

    # prepare nid
    ntype_invmap = {}
    offsets = np.zeros(len(num_nodes) + 1, dtype='int64')
    for i, (ntype, num_nodes_per_type) in enumerate(num_nodes.items()):
        logging.info('Processing node ID for node type %s' % ntype)
        ntype_invmap[ntype] = i
        ntype_dir = os.path.join(output_path, ntype)
        os.makedirs(ntype_dir, exist_ok=True)
        num_nodes_per_partition = \
                (num_nodes_per_type + num_partitions - 1) // num_partitions

        parmetis_nid[ntype] = {
                "format": "csv",
                "data": []}
        weights = [0] * len(num_nodes)
        weights[i] = 1
        for partition_id in range(num_partitions):
            start_id = partition_id * num_nodes_per_partition
            end_id = min(start_id + num_nodes_per_partition, num_nodes_per_type)
            file_path = os.path.join(ntype_dir, '%s-%d.txt' % (ntype, partition_id))
            logging.info('Writing to %s for partition %d for node %d-%d' %
                    (file_path, partition_id, start_id, end_id))

            with open(file_path, 'wb') as f, logging_redirect_tqdm():
                for u in tqdm.trange(start_id, end_id, 1000000):
                    u_end = min(u + 1000000, end_id)
                    buf = np.zeros((u_end - u, 2 + len(num_nodes)), dtype='int64')
                    buf[:, 0] = i
                    buf[:, i] = 1
                    buf[:, -1] = np.arange(u, u_end, dtype='int64')
                    utils.savetxt(f, buf)
            parmetis_nid[ntype]["data"].append([file_path, start_id, end_id])
        offsets[i + 1] = offsets[i] + num_nodes_per_type

    # shuffle and assign partition
    logging.info('Assigning partitions')
    with open(os.path.join(output_path, 'part.id'), 'wb') as f, logging_redirect_tqdm():
        for i in tqdm.trange(0, offsets[-1], 1000000):
            i_end = min(i + 1000000, offsets[-1])
            buf = np.zeros((i_end - i, 2), dtype='int64')
            buf[:, 0] = np.arange(i, i_end, dtype='int64')
            buf[:, 1] = np.random.randint(0, num_partitions, (i_end - i,))
            utils.savetxt(f, buf)
            
    # prepare eid
    etype_invmap = {}
    for i, (cetype, edge_index_metadata) in enumerate(edges.items()):
        logging.info('Processing edge index for edge type %s' % cetype)
        utype, etype, vtype = cetype.split(':')
        etype_invmap[utype, etype, vtype] = i
        cetype_underscore_joined = '___'.join([utype, etype, vtype])
        cetype_path = os.path.join(output_path, cetype_underscore_joined)
        os.makedirs(cetype_path, exist_ok=True)

        parmetis_eid[cetype] = {
                "format": "csv",
                "data": split_edge_index(edge_index_metadata, num_partitions,
                                         cetype_path,
                                         offsets[ntype_invmap[utype]],
                                         offsets[ntype_invmap[vtype]],
                                         cetype_underscore_joined,
                                         i)}

    # prepare node and edge data
    parmetis_ndata = None if ndata is None else _prepare_parmetis_data(output_path, ndata, num_partitions)
    parmetis_edata = None if edata is None else _prepare_parmetis_data(output_path, edata, num_partitions)

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump({
            'nid': parmetis_nid,
            'eid': parmetis_eid,
            'node_data': parmetis_ndata,
            'edge_data': parmetis_edata}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration file')
    parser.add_argument('metadata', help='metadata file')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    basedir = os.path.dirname(args.metadata)
    output_path = os.path.abspath(args.output_path)

    nparts = config['num-partitions']
    with files.setdir(basedir):
        prepare_input(utils.absolutify_metadata(metadata), output_path, nparts, config)
