import numpy as np
import sys

from .csv_dataset import CSVDataset
from .utils import smile_to_bigraph
from ..utils import get_download_dir, download, _get_dgl_url
from ... import backend as F

try:
    import pandas as pd
except ImportError:
    pass

class Tox21(CSVDataset):
    """Tox21 dataset.

    The Toxicology in the 21st Century (https://tripod.nih.gov/tox21/challenge/)
    initiative created a public database measuring toxicity of compounds, which
    has been used in the 2014 Tox21 Data Challenge. The dataset contains qualitative
    toxicity measurements for 8014 compounds on 12 different targets, including nuclear
    receptors and stress response pathways. Each target results in a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for Tox21. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.
    See examples below for more details.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    Parameters
    ----------
    smile_to_graph: callable, str -> DGLGraph
        A function turns smiles into a DGLGraph. Default one can be found
        at python/dgl/data/chem/utils.py named with smile_to_bigraph.
    """
    def __init__(self, smile_to_graph=smile_to_bigraph):
        if 'pandas' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning("Please install pandas")

        self._url = 'dataset/tox21.csv.gz'
        data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(self._url), path=data_path)
        df = pd.read_csv(data_path)
        self.id = df['mol_id']

        df = df.drop(columns=['mol_id'])

        super().__init__(df, smile_to_graph, cache_file_path="tox21_dglgraph.bin")
        self._weight_balancing()

    
    def _weight_balancing(self):
        """Perform re-balancing for each task.

        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        If weight balancing is performed, one attribute will be affected:

        * self._task_pos_weights is set, which is a list of positive sample weights
          for each task.
        """
        num_pos = F.sum(self.labels, dim=0)
        num_indices = F.sum(self.mask, dim=0)
        self._task_pos_weights = (num_indices - num_pos) / num_pos
    

    @property
    def task_pos_weights(self):
        """Get weights for positive samples on each task

        Returns
        -------
        numpy.ndarray
            numpy array gives the weight of positive samples on all tasks
        """
        return self._task_pos_weights
