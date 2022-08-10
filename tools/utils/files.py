import os
from contextlib import contextmanager
import logging
from numpy.lib.format import open_memmap

@contextmanager
def setdir(path):
    try:
        os.makedirs(path, exist_ok=True)
        cwd = os.getcwd()
        logging.info('Changing directory to %s' % path)
        logging.info('Previously: %s' % cwd)
        os.chdir(path)
        yield
    finally:
        logging.info('Restoring directory to %s' % cwd)
        os.chdir(cwd)


def txt2npy(input_path, output_path):
    """Convert a one-dimensional array stored in text, one line per number, into
    a numpy array for fast random access.
    """
    import dask.dataframe as dd
    logging.info('Reading text file %s...' % input_path)
    dask_arr = dd.read_csv(input_path, names=['v'])
    logging.info('Counting lines...')
    dask_arr = dask_arr['v'].to_dask_array(True)    # evaluate shape
    size = dask_arr.shape[0]
    dtype = dask_arr.dtype
    np_arr = open_memmap(output_path, mode='w+', shape=(size,), dtype=dtype)
    logging.info('Saving content to %s...' % output_path)
    dask_arr.store(np_arr)
