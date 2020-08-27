"""Dataset utilities."""
from __future__ import absolute_import

import os
import sys
import hashlib
import warnings
import requests
import pickle
import errno
from multiprocessing import Manager,Process

import numpy as np
import scipy.sparse as sp

try:
    import spacy
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError:
    pass


from .graph_serialize import save_graphs, load_graphs, load_labels
from .tensor_serialize import save_tensors, load_tensors

from .. import backend as F

__all__ = ['loadtxt','download', 'check_sha1', 'extract_archive',
           'get_download_dir', 'Subset', 'split_dataset',
           'save_graphs', "load_graphs", "load_labels", "save_tensors", "load_tensors",
           'parse_word2vec_feature', 'parse_category_single_feat',
           'parse_category_multi_feat', 'parse_numerical_feat',
           'parse_numerical_multihot_feat']

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

################### Feature Processing #######################

def row_normalize(features):
    mx = sp.csr_matrix(features, dtype=np.float32)

    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return np.array(mx.todense())

def col_normalize(features):
    mx = sp.csr_matrix(features, dtype=np.float32)

    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv).transpose()
    mx = mx.dot(c_mat_inv)
    return np.array(mx.todense())

def float_row_l1_normalize(features):
    rowsum = np.sum(np.abs(features), axis=1)
    r_inv = np.power(rowsum, -1).reshape(-1,1)
    r_inv[np.isinf(r_inv)] = 0.
    return features * r_inv

def float_col_l1_normalize(features):
    colsum = np.sum(np.abs(features), axis=0)
    c_inv = np.power(colsum, -1)
    c_inv[np.isinf(c_inv)] = 0.
    return features * c_inv

def float_col_maxmin_normalize(features):
    feats = np.transpose(features)
    min_val = np.reshape(np.amin(feats, axis=1), (-1, 1))
    max_val = np.reshape(np.amax(feats, axis=1), (-1, 1))
    norm = (feats - min_val) / (max_val - min_val)
    norm[np.isnan(norm)] = 0.
    return np.transpose(norm)

def embed_word2vec(str_val, nlps):
    """ Use NLP encoder to encode the string into vector

    There can be multiple NLP encoders in nlps. Each encoder
    is invoded to generate a embedding for the input string and
    the resulting embeddings are concatenated.

    Parameters
    ----------
    str_val : str
        words to encode

    nlps : list of func
        a list of nlp encoder functions
    """
    vector = None
    for nlp in nlps:
        doc = nlp(str_val)
        if vector is None:
            vector = doc.vector
        else:
            vector = np.concatenate((vector, doc.vector))
    return vector

def parse_lang_feat(str_feats, nlp_encoders, verbose=False):
    """ Parse a list of strings using word2vec encoding using NLP encoders in nlps

    Parameters
    ----------
    str_feats : list of str
        list of strings to encode

    nlp_encoders : list of func
        a list of nlp encoder functions

    verbose : bool, optional
        print out debug info
        Default: False

    Return
    ------
    numpy.array
        the encoded features
    """
    features = []
    num_feats = len(str_feats)
    num_process = num_feats if num_feats < 8 else 8 # TODO(xiangsx) get system nproc
    batch_size = (num_feats + num_process - 1) // num_process

    def embed_lang(d, proc_idx, feats):
        res_feats = []
        for s_feat in feats:
            res_feats.append(embed_word2vec(s_feat, nlp_encoders))
        d[proc_idx] = res_feats

    # use multi process to process the feature
    manager = Manager()
    d = manager.dict()
    job=[]
    for i in range(num_process):
        sub_info = str_feats[i * batch_size : (i+1) * batch_size \
                         if (i+1) * batch_size < num_feats else num_feats]
        job.append(Process(target=embed_lang, args=(d, i, sub_info)))

    for p in job:
        p.start()

    for p in job:
        p.join()

    for i in range(num_process):
        if len(d[i]) > 0:
            features.append(d[i])

    features = np.concatenate(features)
    if verbose:
        print(features.shape)

    return features

def parse_word2vec_feature(str_feats, languages, verbose=False):
    """ Parse a list of strings using word2vec encoding using NLP encoders in nlps

    Parameters
    ----------
    str_feats : list of str
        list of strings to encode

    languages : list of string
        list of languages used to encode the feature string.

    verbose : bool, optional
        print out debug info
        Default: False

    Return
    ------
    numpy.array
        the encoded features

    Examples
    --------

    >>> inputs = ['hello', 'world']
    >>> languages = ['en_core_web_lg', 'fr_core_news_lg']
    >>> feats = parse_word2vec_node_feature(inputs, languages)

    """
    import spacy

    nlp_encoders = []
    for lang in languages:
        encoder = spacy.load(lang)
        nlp_encoders.append(encoder)

    return parse_lang_feat(str_feats, nlp_encoders, verbose)

def parse_category_single_feat(category_inputs, norm=None):
    """ Parse categorical features and convert it into onehot encoding.

    Each entity of category_inputs should only contain only one category.

    Parameters
    ----------
    category_inputs : list of str
        input categorical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, sane as None

    Note
    ----
    sklearn.preprocessing.LabelBinarizer is used to convert
    categorical features into a onehot encoding format.

    Return
    ------
    numpy.array
        The features in numpy array

    Examples
    --------

    >>> inputs = ['A', 'B', 'C', 'A']
    >>> feats = parse_category_single_feat(inputs)
    >>> feats
        array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])

    """
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    feat = lb.fit_transform(category_inputs)

    # if there are only 2 catebories,
    # fit_transform only create a array of [0, 1, ...]
    if feat.shape[1] == 1:
        f = np.zeros((feat.shape[0], 2))
        f[range(f.shape[0]),feat.squeeze()] = 1.
        feat = f

    if norm == 'col':
        return col_normalize(feat)
    else:
        return feat

def parse_category_multi_feat(category_inputs, norm=None):
    """ Parse categorical features and convert it into multi-hot encoding.

    Each entity of category_inputs may contain multiple categorical labels.
    It uses multi-hot encoding to encode these labels.

    Parameters
    ----------
    category_inputs : list of list of str
        input categorical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, row-based normalization. Normalize the data for
        each row:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{x_{ij}}}

        Default: None

    Note
    ----
    sklearn.preprocessing.MultiLabelBinarizer is used to convert
    categorical features into a multilabel format.

    Return
    ------
    numpy.array
        The features in numpy array

    Example
    -------

    >>> inputs = [['A', 'B', 'C',], ['A', 'B'], ['C'], ['A']]
    >>> feats = parse_category_multi_feat(inputs)
    >>> feats
        array([[1.,1.,1.],[1.,1.,0.],[0.,0.,1.],[1.,0.,0.]])

    """
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    feat = mlb.fit_transform(category_inputs)

    if norm == 'col':
        return col_normalize(feat)
    if norm == 'row':
        return row_normalize(feat)
    else:
        return feat

def parse_numerical_feat(numerical_inputs, norm=None):
    """ Parse numerical features.

    Parameters
    ----------
    numerical_inputs : list of float or list of list of float
        input numerical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `standard`:, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{|x_{ij}|}}

        (3) `min-max`: column-based min-max normalization. Normalize the data
        for each column:

        .. math::
            norm_i = \frac{x_i - min(x[:])}{max(x[:])-min(x[:])}


    Return
    ------
    numpy.array
        The features in numpy array

    Example

    >>> inputs = [[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]]
    >>> feat = parse_numerical_feat(inputs, norm='col')
    >>> feat
    array([[0.25, 0., 0.],[0.5, 0.33333333, 0.25],[0.25, 0.66666667, -0.75]])

    """
    feat = np.array(numerical_inputs, dtype='float')

    if norm == 'standard':
        return float_col_l1_normalize(feat)
    elif norm == 'min-max':
        return float_col_maxmin_normalize(feat)
    else:
        return feat

def parse_numerical_multihot_feat(input_feats, low, high, bucket_cnt, window_size, norm=None):
    r""" Parse numerical features by matching them into
        different buckets.

    A bucket range based algorithm is used to convert numerical value into multi-hop
    encoding features.

    A numerical value range [low, high) is defined, and it is
    divied into #bucket_cnt buckets. For a input V, we get its effected range as
    [V - window_size/2, V + window_size/2] and check how many buckets it covers in
    [low, high).

    Parameters
    ----------
    input_feats : list of float
        Input numerical features
    low : float
        Lower bound of the range of the numerical values.
        All v_i < low will be set to v_i = low.
    high : float
        Upper bound of the range of the numerical values.
        All v_j > high will be set to v_j = high.
    bucket_cnt: int
        Number of bucket to use.
    slide_window_size: int
        The sliding window used to convert numerical value into bucket number.
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, row-based normalization. Normalize the data for
        each row:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{x_{ij}}}

    Example
    -------

    >>> inputs = [0., 15., 26., 40.]
    >>> low = 10.
    >>> high = 30.
    >>> bucket_cnt = 4
    >>> window_size = 10. # range is 10 ~ 15; 15 ~ 20; 20 ~ 25; 25 ~ 30
    >>> feat = parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size)
    >>> feat
        array([[1., 0., 0., 0],
               [1., 1., 1., 0.],
               [0., 0., 1., 1.],
               [0., 0., 0., 1.]])
    """
    raw_feats = np.array(input_feats, dtype=np.float32)
    num_nodes = raw_feats.shape[0]
    feat = np.zeros((num_nodes, bucket_cnt), dtype=np.float32)

    bucket_size = (high - low) / bucket_cnt
    eposilon = bucket_size / 10
    low_val = raw_feats - window_size/2
    high_val = raw_feats + window_size/2
    low_val[low_val < low] = low
    high_val[high_val < low] = low
    high_val[high_val >= high] = high - eposilon
    low_val[low_val >= high] = high - eposilon
    low_val -= low
    high_val -= low
    low_idx = (low_val / bucket_size).astype('int')
    high_idx = (high_val / bucket_size).astype('int') + 1

    for i in range(raw_feats.shape[0]):
        idx = np.arange(start=low_idx[i], stop=high_idx[i])
        feat[i][idx] = 1.

    if norm == 'col':
        return col_normalize(feat)
    if norm == 'row':
        return row_normalize(feat)
    else:
        return feat
