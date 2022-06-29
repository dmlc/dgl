"""ParMETIS distributed partitioning entry script."""
import json
import os
import argparse

import numpy as np

from ...utils.dispatchers import predicate_dispatch
from .. import files

def _save_partitioned(arr, num_partitions, output_path):
    num_rows = arr.shape[0]
    num_rows_per_partition = (num_rows + num_partitions - 1) // num_rows
    os.makedirs(output_path)
    new_offsets = []
    for i in range(0, num_partitions):
        start = i * num_rows_per_partition
        end = min(num_rows, (i + 1) * num_partitions)
        part_path = os.path.join(output_path, '%d.npy' % i)
        np.save(arr[start:end], part_path)
        new_offsets.append((files.absolutify(part_path), start, end))
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

    num_rows_per_partition = (num_rows + num_partitions - 1) // num_rows
    os.makedirs(output_path)

    part_id = 0
    part = files.np_load(partition_paths[part_id])
    part_offset = 0
    part_remaining = part.shape[0]
    all_offset = 0
    new_offsets = []
    for i in range(0, num_partitions):
        start = i * num_rows_per_partition
        end = min(num_rows, (i + 1) * num_partitions)
        buffer_, part_id, part, part_offset, part_remaining = _get_slice(
                start, end, part_id, part, part_offset, part_remaining)

        part_path = os.path.join(output_path, '%d.npy' % i)
        np.save(buffer_, part_path)
        new_offsets.append((files.absolutify(part_path), start, end))
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

    for partition_id in range(num_partitions):
        start_id = partition_id * num_edges_per_partition
        end_id = min(start_id + num_edges_per_partition, edge_index.shape[1])
        file_path = os.path.join(cetype, '%s-%d.txt' % (cetype, partition_id))
        buffer_ = edge_index[:, start_id:end_id]
        with open(file_path, 'w') as f:
            for j in range(0, end_id - start_id):
                u = buffer_[0, j] + utype_offset
                v = buffer_[1, j] + vtype_offset
                print(u, v, cetype_idx, cetype, file=f)
        result.append([files.absolutify(file_path), start_id, end_id])
    return result

@split_edge_index.register(files.is_numpy)
def _(meta, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
    edge_index = files.np_load(meta['path'])
    return _save_partitioned_edge_index(
            edge_index, num_partitions, output_path, utype_offset, vtype_offset, cetype,
            cetype_idx)

@split_edge_index.register(files.is_c_blob)
def _(meta, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    edge_index = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    return _save_partitioned_edge_index(
            edge_index, num_partitions, output_path, utype_offset, vtype_offset, cetype,
            cetype_idx)

@split_edge_index.register(files.is_partitioned_numpy)
def _(meta, num_partitions, output_path, utype_offset, vtype_offset, cetype, cetype_idx):
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

    num_edges_per_partition = (num_edges + num_partitions - 1) // num_edges
    os.makedirs(output_path)
    output_part_id = 0
    output_part_path = os.path.join(output_path, '%d.txt' % output_part_id)
    output_part_file = open(output_part_path, 'w')
    start_id = row_id = 0
    for i, part_path in enumerate(partition_paths):
        part_edge_index = files.np_load(part_path)
        for j in range(part_edge_index.shape[1]):
            u = part_edge_index[0, j] + utype_offset
            v = part_edge_index[1, j] + vtype_offset
            print(u, v, cetype_idx, cetype, file=output_part_file)
            row_id += 1
            if (row_id % num_edges_per_partition == 0) or (row_id == num_edges):
                output_part_id += 1
                output_part_file.close()
                result.append((files.absolutify(output_part_path), start_id, row_id))
                if row_id < num_edges:
                    output_part_path = os.path.join(output_path, '%d.txt' % output_part_id)
                    output_part_file = open(output_part_path, 'w')
                    start_id = row_id

    return result


def _prepare_parmetis_data(meta, num_partitions):
    parmetis_output = {}
    for i, (ty, type_data) in enumerate(meta.items()):
        os.makedirs(ty, exist_ok=True)
        paths = split_tensor(meta, num_partitions, ty)
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

    os.makedirs(output_path, exist_ok=True)
    cwd = os.getcwd()

    os.chdir(output_path)

    # prepare nid
    ntype_invmap = {}
    offsets = np.zeros(len(num_nodes) + 1, dtype='int64')
    for i, (ntype, num_nodes_per_type) in enumerate(num_nodes.items()):
        ntype_invmap[ntype] = i
        os.makedirs(ntype, exist_ok=True)
        num_nodes_per_partition = \
                (num_nodes_per_type + num_partitions - 1) // num_partitions

        parmetis_nid[ntype] = {
                "format": "csv",
                "data": []}
        for partition_id in range(num_partitions):
            start_id = partition_id * num_nodes_per_partition
            end_id = min(start_id + num_nodes_per_partition, num_nodes_per_type)
            file_path = os.path.join(ntype, '%s-%d.txt' % (ntype, partition_id))
            with open(file_path, 'w') as f:
                for u in range(start_id, end_id):
                    # assuming that the weight is 0
                    print(i, 0, u, file=f)
            parmetis_nid[ntype]["data"].append([file_path, start_id, end_id])
        offsets[i + 1] = offsets[i] + num_nodes_per_type
            
    # prepare eid
    etype_invmap = {}
    for i, (cetype, edge_index_metadata) in enumerate(edges.items()):
        utype, etype, vtype = cetype.split(':')
        etype_invmap[utype, etype, vtype] = i
        cetype_underscore_joined = '___'.join([utype, etype, vtype])
        os.makedirs(cetype_path, exist_ok=True)

        parmetis_eid[cetype] = {
                "format": "csv",
                "data": split_edge_index(edge_index_metadata, num_partitions,
                                         cetype_underscore_joined,
                                         offsets[ntype_invmap[utype]],
                                         offsets[ntype_invmap[vtype]],
                                         cetype_underscore_joined,
                                         i)}

    # prepare node and edge data
    parmetis_ndata = None if ndata is None else _prepare_parmetis_data(ndata)
    parmetis_edata = None if edata is None else _prepare_parmetis_data(edata)

    with open('metadata.json', 'w') as f:
        json.dump({
            'nid': parmetis_nid,
            'eid': parmetis_eid,
            'node_data': parmetis_ndata,
            'edge_data': parmetis_edata}, f)

    os.chdir(cwd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration file')
    parser.add_argument('metadata', help='metadata file')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)

    nparts = config['num-partitions']
    prepare_input(metadata, args.output_path, nparts, config)
