"""Basic DGL Dataset
"""

from __future__ import absolute_import

import os, sys
from .utils import download, extract_archive, get_download_dir, makedirs
from ..utils import retry_method_with_fix

class DGLDataset(object):
    r"""The Basic DGL Dataset for creating graph datasets.

    This class defines a basic template class for DGL Dataset.
    TODO(xiangsx):
        Suport to_pytorch_dataset, to_mxnet_dataset, to_tensorflow_dataset
        so DGL Dataset can easily converted to Framework specific Dataset

    Parameters
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset
    raw_dir : str
        Raw file directory to download/contains the input data.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    """
    def __init__(self, name, url=None, raw_dir=None, force_reload=False):
        self._name = name
        self._url = url
        self._force_reload = force_reload

        # if no dir is provided, the default dgl download dir is used.
        if raw_dir is None:
            self._raw_dir = get_download_dir()

        self._load()
    
    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder.
            Can be ignored if the dataset is already in self.raw_dir
        """
        pass
    
    def save(selfs):
        r"""Save the processed dataset into files.
            Use dgl.utils.data.save_graphs to save dgl graph into files.
            Use dgl.utils.data.save_info to save extra dict information into files.
        """
        pass

    def load(self):
        r"""Load the saved dataset.
            Use dgl.utils.data.load_graphs to load dgl graph from files.
            Use dgl.utils.data.load_info to load extra information into python dict object.
        """
        pass
    
    def process(self, root_path):
        r"""Processes the data under root_path.
            By default root_path = os.path.join(self.raw_dir, self.name).
            One can overwrite raw_path() function to change the path.
        """
        raise NotImplementedError

    def has_cache(self):
        r"""Decide whether there exists a preprocessed dataset
            By default False.
        """
        return False

    @retry_method_with_fix(download)
    def _download(self):
        r"""Download dataset by calling self.download() if the dataset does not exists under self.raw_path.
            By default self.raw_path = os.path.join(self.raw_dir, self.name)
            One can overwrite raw_path() function to change the path.
        """
        if os.path.exists(self.raw_path):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _load(self):
        r"""Entry point from __init__ to load the dataset. 
            if the cache exists:
                Load the dataset from saved dgl graph and information files.
            else:
                1. Download the dataset if needed.
                2. Process the dataset and build the dgl graph.
                3. Save the processed dataset into files.
        """
        if not self._force_reload and self.has_cache():
            self.load()
        else:
            self._download()
            self.process(self.raw_path)
            self.save()

    @property
    def url(self):
        return self._url

    @property
    def name(self):
        return self._name

    @property
    def raw_dir(self):
        return self._raw_dir

    @property
    def raw_path(self):
        return os.path.join(self.raw_dir, self.name)

    @property
    def graph(self):
        raise NotImplementedError

    @property
    def train_mask(self):
        raise NotImplementedError
    
    @property
    def val_mask(self):
        raise NotImplementedError
    
    @property
    def test_mask(self):
        raise NotImplementedError
    
    @property
    def labels(self):
        raise NotImplementedError

    @property
    def num_labels(self):
        raise NotImplementedError
    
    @property
    def node_features(self):
        raise NotImplementedError
    
    @property
    def edge_features(self):
        raise NotImplementedError
    
    @property
    def predict_ntype(self):
        raise NotImplementedError
    
    @property
    def predict_ntid(self):
        raise NotImplementedError
    
    def to_pytorch_dataset(self):
        r"""TODO(xiangsx) Build a torch.utils.data.Dataset from DGLDataset
        """
        raise NotImplementedError
    
    def to_mxnet_dataset(self):
        r"""TODO(xiangsx) Build a mxnet.gluon.data.dataset.Dataset from DGLDataset
        """
        raise NotImplementedError
    
    def to_tensorflow_dataset(self):
        r"""TODO(xiangsx) Build a tensorflow.data.Datasets from DGLDataset
        """
        raise NotImplementedError

class DGLBuiltinDataset(DGLDataset):
    r"""The Basic DGL Dataset Builtin dataset.
        Builtin dataset will be automatically downloaded into ~/.dgl/

    Parameters
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset
    force_reload : bool
        Whether to reload the dataset. Default: False
    """
    def __init__(self, name, url, force_reload=False):
        super(DGLBuiltinDataset, self).__init__(name,
                                                url=url,
                                                raw_dir=None,
                                                force_reload=force_reload)

    def download(self):
        r""" Automatically download data and extract it.
        """
        print(self.url)
        zip_file_path='{}/{}.zip'.format(self.raw_dir, self.name)
        download(self.url, path=zip_file_path)
        extract_archive(zip_file_path, self.raw_path)
