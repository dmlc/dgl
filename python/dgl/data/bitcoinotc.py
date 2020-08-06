""" BitcoinOTC dataset for fraud detection """
import numpy as np
import os
import datetime
import gzip
import shutil

from .dgl_dataset import DGLBuiltinDataset
from .utils import download, makedirs, save_graphs, load_graphs, check_sha1
from ..convert import graph as dgl_graph
from .. import backend as F


class BitcoinOTCDataset(DGLBuiltinDataset):
    r"""BitcoinOTC dataset for fraud detection

    This is who-trusts-whom network of people who trade using Bitcoin on
    a platform called Bitcoin OTC. Since Bitcoin users are anonymous,
    there is a need to maintain a record of users' reputation to prevent
    transactions with fraudulent and risky users.

    Offical website: `<https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html>`_

    Bitcoin OTC dataset statistics:

    - Nodes: 5,881
    - Edges: 35,592
    - Range of edge weight: -10 to +10
    - Percentage of positive edges: 89%

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset.
        Default: False
    verbose: bool
        Whether to print out progress information.
        Default: True.

    Attributes
    ----------
    graphs : list
        A list of DGLGraph objects
    is_temporal : bool
        Indicate whether the graphs are temporal graphs

    Raises
    ------
    UserWarning
        If the raw data is changed in the remote server by the author.

    Examples
    --------
    >>> dataset = BitcoinOTCDataset()
    >>> len(dataset)
    136
    >>> for g in dataset:
    ....    # get edge feature
    ....    edge_weights = g.edata['h']
    ....    # your code here
    >>>
    """

    _url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
    _sha1_str = 'c14281f9e252de0bd0b5f1c6e2bae03123938641'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(BitcoinOTCDataset, self).__init__(name='bitcoinotc',
                                                url=self._url,
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.csv.gz')
        download(self.url, path=gz_file_path)
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name + '.csv.gz'))
        self._extract_gz(gz_file_path, self.raw_path)

    def process(self):
        filename = os.path.join(self.save_path, self.name + '.csv')
        data = np.loadtxt(filename, delimiter=',').astype(np.int64)
        data[:, 0:2] = data[:, 0:2] - data[:, 0:2].min()
        delta = datetime.timedelta(days=14).total_seconds()
        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        time_index = np.around(
            (data[:, 3] - data[:, 3].min()) / delta).astype(np.int64)

        self._graphs = []
        for i in range(time_index.max()):
            row_mask = time_index <= i
            edges = data[row_mask][:, 0:2]
            rate = data[row_mask][:, 2]
            g = dgl_graph((edges[:, 0], edges[:, 1]))
            g.edata['h'] = F.tensor(rate.reshape(-1, 1), dtype=F.data_type_dict['int64'])
            self._graphs.append(g)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self.graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __len__(self):
        r""" Number of graphs in the dataset.

        Return
        -------
        int
        """
        return len(self.graphs)

    def __getitem__(self, item):
        r""" Get graph by index

        Parameters
        ----------
        item : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``edata['h']`` : edge weights
        """
        return self.graphs[item]

    @property
    def is_temporal(self):
        r""" Are the graphs temporal graphs

        Returns
        -------
        bool
        """
        return True

    def _extract_gz(self, file, target_dir, overwrite=False):
        if os.path.exists(target_dir) and not overwrite:
            return
        print('Extracting file to {}'.format(target_dir))
        fname = os.path.basename(file)
        makedirs(target_dir)
        out_file_path = os.path.join(target_dir, fname[:-3])
        with gzip.open(file, 'rb') as f_in:
            with open(out_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


BitcoinOTC = BitcoinOTCDataset
