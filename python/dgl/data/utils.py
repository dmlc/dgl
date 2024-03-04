"""Dataset utilities."""
from __future__ import absolute_import

import errno
import hashlib
import os
import pickle
import sys
import warnings

import networkx.algorithms as A

import numpy as np
import requests
from tqdm.auto import tqdm

from .. import backend as F
from .graph_serialize import load_graphs, load_labels, save_graphs
from .tensor_serialize import load_tensors, save_tensors

__all__ = [
    "loadtxt",
    "download",
    "check_sha1",
    "extract_archive",
    "get_download_dir",
    "Subset",
    "split_dataset",
    "save_graphs",
    "load_graphs",
    "load_labels",
    "save_tensors",
    "load_tensors",
    "add_nodepred_split",
    "add_node_property_split",
    "mask_nodes_by_property",
]


def loadtxt(path, delimiter, dtype=None):
    try:
        import pandas as pd

        df = pd.read_csv(path, delimiter=delimiter, header=None)
        return df.values
    except ImportError:
        warnings.warn(
            "Pandas is not installed, now using numpy.loadtxt to load data, "
            "which could be extremely slow. Accelerate by installing pandas"
        )
        return np.loadtxt(path, delimiter=delimiter)


def _get_dgl_url(file_url):
    """Get DGL online url for download."""
    dgl_repo_url = "https://data.dgl.ai/"
    repo_url = os.environ.get("DGL_REPO", dgl_repo_url)
    if repo_url[-1] != "/":
        repo_url = repo_url + "/"
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
    assert np.allclose(
        np.sum(frac_list), 1.0
    ), "Expect frac_list sum to 1, got {:.4f}".format(np.sum(frac_list))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def download(
    url,
    path=None,
    overwrite=True,
    sha1_hash=None,
    retries=5,
    verify_ssl=True,
    log=True,
):
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
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, (
            "Can't construct file-name from this URL. "
            "Please set the `path` option manually."
        )
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised."
        )

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print("Downloading %s from %s..." % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                # Get the total file size.
                total_size = int(r.headers.get("content-length", 0))
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=fname
                ) as bar:
                    with open(fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                bar.update(len(chunk))
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning(
                        "File {} is downloaded but the content hash does not match."
                        " The repo may be outdated or download may be incomplete. "
                        'If the "repo_url" is overridden, consider switching to '
                        "the default repo.".format(fname)
                    )
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print(
                            "download failed, retrying, {} attempt{} left".format(
                                retries, "s" if retries > 1 else ""
                            )
                        )

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
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def extract_archive(file, target_dir, overwrite=True):
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
    print("Extracting file to {}".format(target_dir))
    if (
        file.endswith(".tar.gz")
        or file.endswith(".tar")
        or file.endswith(".tgz")
    ):
        import tarfile

        with tarfile.open(file, "r") as archive:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(
                tar, path=".", members=None, *, numeric_owner=False
            ):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(archive, path=target_dir)
    elif file.endswith(".gz"):
        import gzip
        import shutil

        with gzip.open(file, "rb") as f_in:
            target_file = os.path.join(target_dir, os.path.basename(file)[:-3])
            with open(target_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif file.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(file, "r") as archive:
            archive.extractall(path=target_dir)
    else:
        raise Exception("Unrecognized file type: " + file)


def get_download_dir():
    """Get the absolute path to the download directory.

    Returns
    -------
    dirname : str
        Path to the download directory
    """
    default_dir = os.path.join(os.path.expanduser("~"), ".dgl")
    dirname = os.environ.get("DGL_DOWNLOAD_DIR", default_dir)
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
    """Save dataset related information into disk.

    Parameters
    ----------
    path : str
        File to save information.
    info : dict
        A python dict storing information to save on disk.
    """
    with open(path, "wb") as pf:
        pickle.dump(info, pf)


def load_info(path):
    """Load dataset related information from disk.

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
    warnings.warn(
        "Property {} will be deprecated, please use {} instead.".format(
            old, new
        )
    )


def deprecate_function(old, new):
    warnings.warn(
        "Function {} will be deprecated, please use {} instead.".format(
            old, new
        )
    )


def deprecate_class(old, new):
    warnings.warn(
        "Class {} will be deprecated, please use {} instead.".format(old, new)
    )


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
    assert isinstance(mask, np.ndarray), (
        "input for generate_mask_tensor" "should be an numpy ndarray"
    )
    if F.backend_name == "mxnet":
        return F.tensor(mask, dtype=F.data_type_dict["float32"])
    else:
        return F.tensor(mask, dtype=F.data_type_dict["bool"])


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


def add_nodepred_split(dataset, ratio, ntype=None):
    """Split the given dataset into training, validation and test sets for
    transductive node predction task.

    It adds three node mask arrays ``'train_mask'``, ``'val_mask'`` and ``'test_mask'``,
    to each graph in the dataset. Each sample in the dataset thus must be a :class:`DGLGraph`.

    Fix the random seed of NumPy to make the result deterministic::

        numpy.random.seed(42)

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to modify.
    ratio : (float, float, float)
        Split ratios for training, validation and test sets. Must sum to one.
    ntype : str, optional
        The node type to add mask for.

    Examples
    --------
    >>> dataset = dgl.data.AmazonCoBuyComputerDataset()
    >>> print('train_mask' in dataset[0].ndata)
    False
    >>> dgl.data.utils.add_nodepred_split(dataset, [0.8, 0.1, 0.1])
    >>> print('train_mask' in dataset[0].ndata)
    True
    """
    if len(ratio) != 3:
        raise ValueError(
            f"Split ratio must be a float triplet but got {ratio}."
        )
    for i in range(len(dataset)):
        g = dataset[i]
        n = g.num_nodes(ntype)
        idx = np.arange(0, n)
        np.random.shuffle(idx)
        n_train, n_val, n_test = (
            int(n * ratio[0]),
            int(n * ratio[1]),
            int(n * ratio[2]),
        )
        train_mask = generate_mask_tensor(idx2mask(idx[:n_train], n))
        val_mask = generate_mask_tensor(
            idx2mask(idx[n_train : n_train + n_val], n)
        )
        test_mask = generate_mask_tensor(idx2mask(idx[n_train + n_val :], n))
        g.nodes[ntype].data["train_mask"] = train_mask
        g.nodes[ntype].data["val_mask"] = val_mask
        g.nodes[ntype].data["test_mask"] = test_mask


def mask_nodes_by_property(property_values, part_ratios, random_seed=None):
    """Provide the split masks for a node split with distributional shift based on a given
    node property, as proposed in `Evaluating Robustness and Uncertainty of Graph Models
    Under Structural Distributional Shifts <https://arxiv.org/abs/2302.13875>`__

    It considers the in-distribution (ID) and out-of-distribution (OOD) subsets of nodes.
    The ID subset includes training, validation and testing parts, while the OOD subset
    includes validation and testing parts. It sorts the nodes in the ascending order of
    their property values, splits them into 5 non-intersecting parts, and creates 5
    associated node mask arrays:
        - 3 for the ID nodes: ``'in_train_mask'``, ``'in_valid_mask'``, ``'in_test_mask'``,
        - and 2 for the OOD nodes: ``'out_valid_mask'``, ``'out_test_mask'``.

    Parameters
    ----------
    property_values : numpy ndarray
        The node property (float) values by which the dataset will be split.
        The length of the array must be equal to the number of nodes in graph.
    part_ratios : list
        A list of 5 ratios for training, ID validation, ID test,
        OOD validation, OOD testing parts. The values in the list must sum to one.
    random_seed : int, optional
        Random seed to fix for the initial permutation of nodes. It is
        used to create a random order for the nodes that have the same
        property values or belong to the ID subset. (default: None)

    Returns
    ----------
    split_masks : dict
        A python dict storing the mask names as keys and the corresponding
        node mask arrays as values.

    Examples
    --------
    >>> num_nodes = 1000
    >>> property_values = np.random.uniform(size=num_nodes)
    >>> part_ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
    >>> split_masks = dgl.data.utils.mask_nodes_by_property(property_values, part_ratios)
    >>> print('in_valid_mask' in split_masks)
    True
    """

    num_nodes = len(property_values)
    part_sizes = np.round(num_nodes * np.array(part_ratios)).astype(int)
    part_sizes[-1] -= np.sum(part_sizes) - num_nodes

    generator = np.random.RandomState(random_seed)
    permutation = generator.permutation(num_nodes)

    node_indices = np.arange(num_nodes)[permutation]
    property_values = property_values[permutation]
    in_distribution_size = np.sum(part_sizes[:3])

    node_indices_ordered = node_indices[np.argsort(property_values)]
    node_indices_ordered[:in_distribution_size] = generator.permutation(
        node_indices_ordered[:in_distribution_size]
    )

    sections = np.cumsum(part_sizes)
    node_split = np.split(node_indices_ordered, sections)[:-1]
    mask_names = [
        "in_train_mask",
        "in_valid_mask",
        "in_test_mask",
        "out_valid_mask",
        "out_test_mask",
    ]
    split_masks = {}

    for mask_name, node_indices in zip(mask_names, node_split):
        split_mask = idx2mask(node_indices, num_nodes)
        split_masks[mask_name] = generate_mask_tensor(split_mask)

    return split_masks


def add_node_property_split(
    dataset, part_ratios, property_name, ascending=True, random_seed=None
):
    """Create a node split with distributional shift based on a given node property,
    as proposed in `Evaluating Robustness and Uncertainty of Graph Models Under
    Structural Distributional Shifts <https://arxiv.org/abs/2302.13875>`__

    It splits the nodes of each graph in the given dataset into 5 non-intersecting
    parts based on their structural properties. This can be used for transductive node
    prediction task with distributional shifts.

    It considers the in-distribution (ID) and out-of-distribution (OOD) subsets of nodes.
    The ID subset includes training, validation and testing parts, while the OOD subset
    includes validation and testing parts. As a result, it creates 5 associated node mask
    arrays for each graph:
        - 3 for the ID nodes: ``'in_train_mask'``, ``'in_valid_mask'``, ``'in_test_mask'``,
        - and 2 for the OOD nodes: ``'out_valid_mask'``, ``'out_test_mask'``.

    This function implements 3 particular strategies for inducing distributional shifts
    in graph — based on **popularity**, **locality** or **density**.

    Parameters
    ----------
    dataset : :class:`~DGLDataset` or list of :class:`~dgl.DGLGraph`
        The dataset to induce structural distributional shift.
    part_ratios : list
        A list of 5 ratio values for training, ID validation, ID test,
        OOD validation and OOD test parts. The values must sum to 1.0.
    property_name : str
        The name of the node property to be used, which must be
        ``'popularity'``, ``'locality'`` or ``'density'``.
    ascending : bool, optional
        Whether to sort nodes in the ascending order of the node property,
        so that nodes with greater values of the property are considered
        to be OOD (default: True)
    random_seed : int, optional
        Random seed to fix for the initial permutation of nodes. It is
        used to create a random order for the nodes that have the same
        property values or belong to the ID subset. (default: None)

    Examples
    --------
    >>> dataset = dgl.data.AmazonCoBuyComputerDataset()
    >>> print('in_valid_mask' in dataset[0].ndata)
    False
    >>> part_ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
    >>> property_name = 'popularity'
    >>> dgl.data.utils.add_node_property_split(dataset, part_ratios, property_name)
    >>> print('in_valid_mask' in dataset[0].ndata)
    True
    """

    assert property_name in [
        "popularity",
        "locality",
        "density",
    ], "The name of property has to be 'popularity', 'locality', or 'density'"

    assert len(part_ratios) == 5, "part_ratios must contain 5 values"

    import networkx as nx

    for idx in range(len(dataset)):
        graph_dgl = dataset[idx]
        graph_nx = nx.Graph(graph_dgl.to_networkx())

        compute_property_fn = _property_name_to_compute_fn[property_name]
        property_values = compute_property_fn(graph_nx, ascending)

        node_masks = mask_nodes_by_property(
            property_values, part_ratios, random_seed
        )

        for mask_name, node_mask in node_masks.items():
            graph_dgl.ndata[mask_name] = node_mask


def _compute_popularity_property(graph_nx, ascending=True):
    direction = -1 if ascending else 1
    property_values = direction * np.array(list(A.pagerank(graph_nx).values()))
    return property_values


def _compute_locality_property(graph_nx, ascending=True):
    num_nodes = graph_nx.number_of_nodes()
    pagerank_values = np.array(list(A.pagerank(graph_nx).values()))

    personalization = dict(zip(range(num_nodes), [0.0] * num_nodes))
    personalization[np.argmax(pagerank_values)] = 1.0

    direction = -1 if ascending else 1
    property_values = direction * np.array(
        list(A.pagerank(graph_nx, personalization=personalization).values())
    )
    return property_values


def _compute_density_property(graph_nx, ascending=True):
    direction = -1 if ascending else 1
    property_values = direction * np.array(
        list(A.clustering(graph_nx).values())
    )
    return property_values


_property_name_to_compute_fn = {
    "popularity": _compute_popularity_property,
    "locality": _compute_locality_property,
    "density": _compute_density_property,
}
