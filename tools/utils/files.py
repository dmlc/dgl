import copy
import os
import numpy as np
import shutil
import pathlib
from contextlib import contextmanager
import logging

def get_partitioned_numpy_file_list(path):
    numpy_file_list = {}
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            try:
                id_ = int(filename[:-4])
                numpy_file_list[id_] = filename
            except ValueError:
                continue
    ids, filenames = zip(*numpy_file_list.items())
    ids = np.asarray(ids)
    indices = np.argsort(ids)
    filenames = [filenames[i] for i in indices]
    ids = ids[indices]
    
    if not np.array_equal(ids, np.arange(len(numpy_file_list))):
        raise RuntimeError(
                'Expected the file names as consecutive integers (e.g. 0.npy, 1.npy, ...)')
    return filenames

def is_numpy(meta):
    """Check if a metadata describes a numpy tensor."""
    return meta.get('fmt', '') == 'numpy' or meta.get('path', '').endswith('.npy')

def is_c_blob(meta):
    """Check whether a metadata properly describes a C-array blob."""
    if meta.get('fmt', '') == 'blob':
        if 'dtype' not in meta or 'shape' not in 'meta':
            raise ValueError('Requires "dtype" and "shape" in metadata when "fmt" is "blob".')
        return True
    return False

def is_partitioned_numpy(meta):
    """Check whether a metadata properly describes a partitioned numpy array."""
    return meta.get('fmt', '') == 'partitioned-numpy'

def np_load(path):
    if os.path.getsize(path) >= 1000000000:
        # Use mmap for files larger than 1GB
        return np.load(path, mmap_mode='r')
    else:
        return np.load(path)

@contextmanager
def setdir(path):
    try:
        cwd = os.getcwd()
        logging.info('Changing directory to %s' % path)
        logging.info('Previously: %s' % cwd)
        os.chdir(path)
        yield
    finally:
        logging.info('Restoring directory to %s' % cwd)
        os.chdir(cwd)

def copypath(src, dst):
    logging.info('Copying from %s to %s' % (src, dst))
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)

def copypath_with_suffix(src, dst):
    src_path = pathlib.Path(src)
    dst_path = pathlib.Path(dst).with_suffix(src_path.suffix)
    copypath(str(src_path), str(dst_path))
    return str(dst_path)
