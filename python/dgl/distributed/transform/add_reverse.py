import os
import json
import shutil
import copy
import pathlib
import numpy as np

from ...utils.dispatchers import predicate_dispatch
from .. import files

@predicate_dispatch
def reverse(meta, output_path):
    """Saves the reversed edge index into the output path.
    
    Returns the metadata entry of the output.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta.get('fmt'))

@reverse.register(files.is_numpy)
def _(meta, output_path):
    arr = files.np_load(path)
    new_path = str(pathlib.Path(output_path).with_suffix('.npy'))
    np.save(arr[[1, 0]], new_path)
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
        np.save(os.join(output_path, filename), arr[[1, 0]])
    return {'fmt': 'partitioned-numpy', 'path': output_path}

@predicate_dispatch
def append_reverse(meta, output_path):
    """Append the reversed edge index to the given edge index, and save the result
    to the output path.
    
    Returns the format of output.
    """
    raise RuntimeError("Don't know how to resolve format: %s" % meta.get('fmt'))

@append_reverse.register(files.is_numpy)
def _(meta, output_path):
    path = meta['path']
    arr = files.np_load(path)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(path, '0.npy'), arr)
    np.save(os.path.join(path, '1.npy'), arr[[1, 0]])
    return {'fmt': 'partitioned-numpy', 'path': output_path}

@append_reverse.register(files.is_c_blob)
def _(meta, output_path):
    path = meta['path']
    dtype = meta['dtype']
    shape = [int(_) for _ in meta['shape'].split(',')]
    arr = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    new_path = str(pathlib.Path(output_path).with_suffix('.dat'))
    new_arr = np.memmap(new_path, mode='w+', dtype=dtype, shape=(shape[0], shape[1] * 2))
    new_arr[:, :shape[1]] = arr
    new_arr[1, shape[1]:] = arr[0]
    new_arr[0, shape[1]:] = arr[1]
    return {'fmt': 'blob', 'path': output_path, 'dtype': new_arr.dtype.name,
            'shape': list(new_arr.shape)}

@append_reverse.register(files.is_partitioned_numpy)
def _(meta, output_path):
    path = meta['path']
    filenames = files.get_partitioned_numpy_file_list(path)
    os.makedirs(output_path, exist_ok=True)
    for i, filename in enumerate(filenames):
        old_path = os.path.join(path, filename)
        new_path = os.path.join(output_path, '%d.npy' % i)
        shutil.copy(old_path, new_path)
    for i, filename in enumerate(self.filenames):
        old_path = os.path.join(path, filename)
        new_path = os.path.join(output_path, '%d.npy' % (i + len(filenames)))
        arr = files.np_load(old_path)
        np.save(new_path, arr)
    return {'fmt': 'partitioned-numpy', 'path': output_path}

def add_reverse(metadata, output_path, config):
    new_metadata = copy.deepcopy(metadata)
    edges = metadata.get('edges', None)
    cwd = os.getcwd()

    os.makedirs(output_path, exist_ok=True)
    os.chdir(output_path)

    for cetype, rev_cetype in config.items():
        cetype_path = cetype.replace(':', '___')
        rev_cetype_path = rev_cetype.replace(':', '___')

        if cetype == rev_cetype:
            os.makedirs(cetype, exist_ok=True)
            new_metadata['edges'][cetype] = append_reverse(edges[cetype], cetype_path)
        else:
            os.makedirs(cetype, exist_ok=True)
            os.makedirs(rev_cetype, exist_ok=True)
            new_metadata['edges'][cetype] = edges[cetype]["path"]
            new_metadata['edges'][rev_cetype] = reverse(edges[cetype], rev_cetype_path)
            if os.path.isdir(edges[cetype]["path"]):
                shutil.copytree(edges[cetype]["path"], cetype_path, dirs_exist_ok=True)
            else:
                shutil.copy(edges[cetype]["path"], cetype_path)

    with open('metadata.json', 'w') as f:
        json.dump(new_metadata, f)
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

    add_reverse(metadata, args.output_path, nparts, config)
