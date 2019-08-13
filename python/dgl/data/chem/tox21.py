import numpy as np

from .csv_dataset import CSVDataset
from .utils import smile2graph
from ..utils import get_download_dir, download, _get_dgl_url, Subset

try:
    import pandas as pd
except ImportError:
    raise ImportError("Please install pandas")


class Tox21(CSVDataset):
    _urls = 'dataset/tox21.csv.gz'

    def __init__(self, smile2graph=smile2graph):
        data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(self._urls), path=data_path)
        df = pd.read_csv(data_path)
        df = df.drop(columns=['mol_id'])
        super().__init__(df, smile2graph, cache_file_path="tox21_dglgraph.pkl")
        self._weight_balancing()

    def _weight_balancing(self):
        num_pos = np.sum(self.labels, axis=0)
        num_indices = np.sum(self.mask, axis=0)
        self._task_pos_weights = (num_indices - num_pos) / num_pos

    @property
    def task_pos_weights(self):
        return self._task_pos_weights