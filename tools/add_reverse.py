import os
import json
import shutil
import copy
import pathlib
import argparse
import logging
import numpy as np
from numpy.lib.format import open_memmap

from utils.dispatchers import predicate_dispatch
from utils import files
import utils

logging.basicConfig(level='INFO')

@predicate_dispatch
def reverse(meta, output_path):
    """Saves the reversed edge index into the output path.
    
    Returns the metadata entry of the output.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta)

@reverse.register(files.is_numpy)
def _(meta, output_path):
    path = meta['path']
    arr = files.np_load(path)
    new_path = str(pathlib.Path(output_path).with_suffix('.npy'))
    new_arr = open_memmap(new_path, mode='w+', dtype=arr.dtype, shape=arr.shape)
    new_arr[1] = arr[0]
    new_arr[0] = arr[1]
    return {'fmt': 'numpy', 'path': new_path}

@reverse.register(files.is_c_blob)
def _(meta, output_path):
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    arr = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    new_path = str(pathlib.Path(output_path).with_suffix('.dat'))
    new_arr = np.memmap(new_path, mode='w+', dtype=arr.dtype, shape=arr.shape)
    new_arr[0] = arr[1]
    new_arr[1] = arr[0]
    return {'fmt': 'blob', 'path': new_path, 'dtype': new_arr.dtype.name,
            'shape': list(arr.shape)}

@reverse.register(files.is_partitioned_numpy)
def _(meta, output_path):
    path = meta['path']
    filenames = files.get_partitioned_numpy_file_list(path)
    os.makedirs(output_path, exist_ok=True)
    for filename in filenames:
        arr = files.np_load(os.join(path, filename))
        new_arr = open_memmap(os.join(output_path, filename), mode='w+', dtype=arr.dtype, shape=arr.shape)
        new_arr[0] = arr[1]
        new_arr[1] = arr[0]
    return {'fmt': 'partitioned-numpy', 'path': output_path}

@predicate_dispatch
def append_reverse(meta, output_path):
    """Append the reversed edge index to the given edge index, and save the result
    to the output path.
    
    Returns the format of output.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta)

@append_reverse.register(files.is_numpy)
def _(meta, output_path):
    logging.info('Appending reverse edge index for metadata %s into %s' % (meta, output_path))
    path = meta['path']
    arr = files.np_load(path)
    os.makedirs(output_path, exist_ok=True)
    new_arr0 = open_memmap(os.path.join(output_path, '0.npy'), mode='w+', dtype=arr.dtype, shape=arr.shape)
    new_arr1 = open_memmap(os.path.join(output_path, '1.npy'), mode='w+', dtype=arr.dtype, shape=arr.shape)
    new_arr0[:] = arr[:]
    new_arr1[0] = arr[1]
    new_arr1[1] = arr[0]
    output_meta = {'fmt': 'partitioned-numpy', 'path': output_path}
    logging.info('Output metadata: %s' % output_meta)
    return output_meta

@append_reverse.register(files.is_c_blob)
def _(meta, output_path):
    logging.info('Appending reverse edge index for metadata %s into %s' % (meta, output_path))
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    arr = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    new_path = str(pathlib.Path(output_path).with_suffix('.dat'))
    new_arr = np.memmap(new_path, mode='w+', dtype=dtype, shape=(shape[0], shape[1] * 2))
    new_arr[:, :shape[1]] = arr
    new_arr[1, shape[1]:] = arr[0]
    new_arr[0, shape[1]:] = arr[1]
    output_meta = {'fmt': 'blob', 'path': output_path, 'dtype': new_arr.dtype.name,
            'shape': list(new_arr.shape)}
    logging.info('Output metadata: %s' % output_meta)
    return output_meta

@append_reverse.register(files.is_partitioned_numpy)
def _(meta, output_path):
    logging.info('Appending reverse edge index for metadata %s into %s' % (meta, output_path))
    path = meta['path']
    filenames = files.get_partitioned_numpy_file_list(path)
    os.makedirs(output_path, exist_ok=True)
    for i, filename in enumerate(filenames):
        old_path = os.path.join(path, filename)
        new_path = os.path.join(output_path, '%d.npy' % i)
        files.copypath(old_path, new_path)
    for i, filename in enumerate(self.filenames):
        old_path = os.path.join(path, filename)
        new_path = os.path.join(output_path, '%d.npy' % (i + len(filenames)))
        logging.info('Reversing %s to %s in-memory' % (old_path, new_path))
        arr = files.np_load(old_path)
        new_arr = open_memmap(new_path, mode='w+', dtype=arr.dtype, shape=arr.shape)
        new_arr[1] = arr[0]
        new_arr[0] = arr[1]
    output_meta = {'fmt': 'partitioned-numpy', 'path': output_path}
    logging.info('Output metadata: %s' % output_meta)
    return output_meta

@predicate_dispatch
def append_reverse_edata(meta, output_path):
    """Append the reversed edge index to the given edge index, and save the result
    to the output path.
    
    Returns the format of output.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta)

@append_reverse_edata.register(files.is_numpy)
def _(meta, output_path):
    logging.info('Appending reverse edge data for metadata %s into %s' % (meta, output_path))
    path = meta['path']
    arr = files.np_load(path)
    os.makedirs(output_path, exist_ok=True)
    files.copypath(path, os.path.join(output_path, '0.npy'))
    files.copypath(path, os.path.join(output_path, '1.npy'))
    output_meta = {'fmt': 'partitioned-numpy', 'path': output_path}
    logging.info('Output metadata: %s' % output_meta)
    return output_meta

@append_reverse_edata.register(files.is_c_blob)
def _(meta, output_path):
    logging.info('Appending reverse edge data for metadata %s into %s' % (meta, output_path))
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    arr = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    new_path = str(pathlib.Path(output_path).with_suffix('.dat'))
    new_arr = np.memmap(new_path, mode='w+', dtype=dtype, shape=(shape[0] * 2, *shape[1:]))
    new_arr[:shape[0]] = arr[:]
    new_arr[shape[0]:] = arr[:]
    output_meta = {'fmt': 'blob', 'path': output_path, 'dtype': new_arr.dtype.name,
            'shape': list(new_arr.shape)}
    logging.info('Output metadata: %s' % output_meta)
    return output_meta

@append_reverse_edata.register(files.is_partitioned_numpy)
def _(meta, output_path):
    logging.info('Appending reverse edge data for metadata %s into %s' % (meta, output_path))
    path = meta['path']
    filenames = files.get_partitioned_numpy_file_list(path)
    os.makedirs(output_path, exist_ok=True)
    for i, filename in enumerate(filenames):
        old_path = os.path.join(path, filename)
        new_path = os.path.join(output_path, '%d.npy' % i)
        new_path_rev = os.path.join(output_path, '%d.npy' % (i + len(filenames)))
        files.copypath(old_path, new_path)
        files.copypath(old_path, new_path_rev)
    output_meta = {'fmt': 'partitioned-numpy', 'path': output_path}
    logging.info('Output metadata: %s' % output_meta)
    return output_meta

def add_reverse(metadata, output_path, config):
    new_metadata = copy.deepcopy(metadata)
    edges = metadata.get('edges', {})
    edata = metadata.get('edge_data', {})
    cetype_dict = utils.get_cetype_dict_from_meta(edges)

    os.makedirs(output_path, exist_ok=True)

    for cetype, rev_cetype in config.items():
        utype, etype, vtype = cetype = utils.get_cetype(cetype_dict, cetype)
        cetype = utils.join_cetype(cetype)
        rev_cetype = utils.join_cetype((vtype, rev_cetype, utype))
        cetype_path = os.path.join(output_path, utils.join_cetype_path(cetype))
        rev_cetype_path = os.path.join(output_path, utils.join_cetype_path(rev_cetype))

        if cetype == rev_cetype:
            logging.info('Adding reverse edges on the same type %s' % cetype)
            utils.set_etype_metadata(
                    new_metadata['edges'], cetype,
                    append_reverse(edges[cetype], cetype_path))
            for key, feat_meta in utils.find_etype_metadata(edata, cetype).items():
                logging.info('Adding reverse edge data on type %s for key %s with metadata %s' % (
                    cetype, key, feat_meta))
                cetype_key_path = os.path.join(cetype_path, key)
                cetype_edata = utils.find_etype_metadata(new_metadata['edge_data'], cetype)
                cetype_edata[key] = append_reverse_edata(feat_meta, cetype_key_path)
        else:
            logging.info('Creating reverse edges on different type %s for %s' % (rev_cetype, cetype))
            utils.set_etype_metadata(
                    new_metadata['edges'], rev_cetype,
                    reverse(edges[cetype], rev_cetype_path))
            cetype_edge_index_path = files.copypath_with_suffix(edges[cetype]["path"], cetype_path)
            cetype_edge_index = utils.find_etype_metadata(new_metadata['edges'], cetype)
            cetype_edge_index["path"] = cetype_edge_index_path

            # copy edata
            for key, feat_meta in utils.find_etype_metadata(edata, cetype).items():
                cetype_key_path = os.path.join(cetype_path, key)
                cetype_key_path = files.copypath_with_suffix(feat_meta["path"], cetype_key_path)
                cetype_edata = utils.find_etype_metadata(new_metadata['edge_data'], cetype)
                cetype_edata[key]["path"] = cetype_key_path

                # reverse edge type is always new so we don't have to call
                # find_etype_metadata to find existing etype keys from canonical etype.
                rev_cetype_key_path = os.path.join(rev_cetype_path, key)
                rev_cetype_key_path = files.copypath_with_suffix(feat_meta["path"], rev_cetype_key_path)
                new_metadata['edge_data'][rev_cetype] = cetype_edata.copy()
                new_metadata['edge_data'][rev_cetype][key]["path"] = rev_cetype_key_path

    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(new_metadata, f)

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

    with files.setdir(basedir):
        add_reverse(utils.absolutify_metadata(metadata), output_path, config)
