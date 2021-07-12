"""Dataset utilities."""
from __future__ import absolute_import

import os
import sys
import hashlib
import warnings
import requests
import pickle
import errno
import numpy as np

import pickle
import errno

from .graph_serialize import save_graphs, load_graphs, load_labels
from .tensor_serialize import save_tensors, load_tensors

from .. import backend as F

__all__ = ['loadtxt','download', 'check_sha1', 'extract_archive',
           'get_download_dir', 'Subset', 'split_dataset',
           'save_graphs', "load_graphs", "load_labels", "save_tensors", "load_tensors"]

def loadtxt(path, delimiter, dtype=None):
    try:
        import pandas as pd
        df = pd.read_csv(path, delimiter=delimiter, header=None)
        return df.values
    except ImportError:
        warnings.warn("Pandas is not installed, now using numpy.loadtxt to load data, "
                        "which could be extremely slow. Accelerate by installing pandas")
        return np.loadtxt(path, delimiter=delimiter)

def _get_dgl_url(file_url):
    """Get DGL online url for download."""
    dgl_repo_url = 'https://data.dgl.ai/'
    repo_url = os.environ.get('DGL_REPO', dgl_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    return repo_url + file_url


def split_dataset(dataset, frac_list=None, shuffle=False, random_state=None):
    """Split dataset into training, validation and test set.

    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the number of datapoints and ``dataset[i]``
        gives the ith datapoint.
    frac_list : list or None, optional
        A list of length 3 containing the fraction to use for training,
        validation and test. If None, we will use [0.8, 0.1, 0.1].
    shuffle : bool, optional
        By default we perform a consecutive split of the dataset. If True,
        we will first randomly shuffle the dataset.
    random_state : None, int or array_like, optional
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive, an array
        (or other sequence) of such integers, or None (the default).
        If seed is None, then RandomState will try to read data from /dev/urandom
        (or the Windows analogue) if available or seed from the clock otherwise.

    Returns
    -------
    list of length 3
        Subsets for training, validation and test.
    """
    from itertools import accumulate
    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    frac_list = np.asarray(frac_list)
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    if shuffle:
        indices = np.random.RandomState(
            seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]


def download(url, path=None, overwrite=True, sha1_hash=None, retries=5, verify_ssl=True, log=True):
    """Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with the same name as in url.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
        By default always overwrites the downloaded file.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print('Downloading %s from %s...' % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'
                                      ' The repo may be outdated or download may be incomplete. '
                                      'If the "repo_url" is overridden, consider switching to '
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    return fname


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def extract_archive(file, target_dir, overwrite=False):
    """Extract archive file.

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    overwrite : bool, default True
        Whether to overwrite the contents inside the directory.
        By default always overwrites.
    """
    if os.path.exists(target_dir) and not overwrite:
        return
    print('Extracting file to {}'.format(target_dir))
    if file.endswith('.tar.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        import tarfile
        with tarfile.open(file, 'r') as archive:
            archive.extractall(path=target_dir)
    elif file.endswith('.gz'):
        import gzip
        import shutil
        with gzip.open(file, 'rb') as f_in:
            with open(file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif file.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file, 'r') as archive:
            archive.extractall(path=target_dir)
    else:
        raise Exception('Unrecognized file type: ' + file)


def get_download_dir():
    """Get the absolute path to the download directory.

    Returns
    -------
    dirname : str
        Path to the download directory
    """
    default_dir = os.path.join(os.path.expanduser('~'), '.dgl')
    dirname = os.environ.get('DGL_DOWNLOAD_DIR', default_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e

def save_info(path, info):
    """ Save dataset related information into disk.

    Parameters
    ----------
    path : str
        File to save information.
    info : dict
        A python dict storing information to save on disk.
    """
    with open(path, "wb" ) as pf:
        pickle.dump(info, pf)


def load_info(path):
    """ Load dataset related information from disk.

    Parameters
    ----------
    path : str
        File to load information from.

    Returns
    -------
    info : dict
        A python dict storing information loaded from disk.
    """
    with open(path, "rb") as pf:
        info = pickle.load(pf)
    return info

def deprecate_property(old, new):
    warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))


def deprecate_function(old, new):
    warnings.warn('Function {} will be deprecated, please use {} instead.'.format(old, new))


def deprecate_class(old, new):
    warnings.warn('Class {} will be deprecated, please use {} instead.'.format(old, new))

def idx2mask(idx, len):
    """Create mask."""
    mask = np.zeros(len)
    mask[idx] = 1
    return mask

def generate_mask_tensor(mask):
    """Generate mask tensor according to different backend
    For torch and tensorflow, it will create a bool tensor
    For mxnet, it will create a float tensor
    Parameters
    ----------
    mask: numpy ndarray
        input mask tensor
    """
    assert isinstance(mask, np.ndarray), "input for generate_mask_tensor" \
        "should be an numpy ndarray"
    if F.backend_name == 'mxnet':
        return F.tensor(mask, dtype=F.data_type_dict['float32'])
    else:
        return F.tensor(mask, dtype=F.data_type_dict['bool'])

class Subset(object):
    """Subset of a dataset at specified indices

    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        return self.dataset[self.indices[item]]

    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)
