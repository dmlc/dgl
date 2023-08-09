""" GDELT dataset for temporal graph """
import os

import numpy as np

from .. import backend as F
from ..convert import graph as dgl_graph
from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, load_info, loadtxt, save_info


class GDELTDataset(DGLBuiltinDataset):
    r"""GDELT dataset for event-based temporal graph

    The Global Database of Events, Language, and Tone (GDELT) dataset.
    This contains events happend all over the world (ie every protest held
    anywhere in Russia on a given day is collapsed to a single entry).
    This Dataset consists ofevents collected from 1/1/2018 to 1/31/2018
    (15 minutes time granularity).

    Reference:

        - `Recurrent Event Network for Reasoning over Temporal Knowledge Graphs <https://arxiv.org/abs/1904.05530>`_
        - `The Global Database of Events, Language, and Tone (GDELT) <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075>`_

    Statistics:

    - Train examples: 2,304
    - Valid examples: 288
    - Test examples: 384

    Parameters
    ----------
    mode : str
        Must be one of ('train', 'valid', 'test'). Default: 'train'
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    start_time : int
        Start time of the temporal graph
    end_time : int
        End time of the temporal graph
    is_temporal : bool
        Does the dataset contain temporal graphs

    Examples
    ----------
    >>> # get train, valid, test dataset
    >>> train_data = GDELTDataset()
    >>> valid_data = GDELTDataset(mode='valid')
    >>> test_data = GDELTDataset(mode='test')
    >>>
    >>> # length of train set
    >>> train_size = len(train_data)
    >>>
    >>> for g in train_data:
    ....    e_feat = g.edata['rel_type']
    ....    # your code here
    ....
    >>>
    """

    def __init__(
        self,
        mode="train",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        mode = mode.lower()
        assert mode in ["train", "valid", "test"], "Mode not valid."
        self.mode = mode
        self.num_nodes = 23033
        _url = _get_dgl_url("dataset/gdelt.zip")
        super(GDELTDataset, self).__init__(
            name="GDELT",
            url=_url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        file_path = os.path.join(self.raw_path, self.mode + ".txt")
        self.data = loadtxt(file_path, delimiter="\t").astype(np.int64)

        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        self.time_index = np.floor(self.data[:, 3] / 15).astype(np.int64)
        self._start_time = self.time_index.min()
        self._end_time = self.time_index.max()

    @property
    def info_path(self):
        return os.path.join(self.save_path, self.mode + "_info.pkl")

    def has_cache(self):
        return os.path.exists(self.info_path)

    def save(self):
        save_info(
            self.info_path,
            {
                "data": self.data,
                "time_index": self.time_index,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
        )

    def load(self):
        info = load_info(self.info_path)
        self.data, self.time_index, self._start_time, self._end_time = (
            info["data"],
            info["time_index"],
            info["start_time"],
            info["end_time"],
        )

    @property
    def start_time(self):
        r"""Start time of events in the temporal graph

        Returns
        -------
        int
        """
        return self._start_time

    @property
    def end_time(self):
        r"""End time of events in the temporal graph

        Returns
        -------
        int
        """
        return self._end_time

    def __getitem__(self, t):
        r"""Get graph by with events before time `t + self.start_time`

        Parameters
        ----------
        t : int
            Time, its value must be in range [0, `self.end_time` - `self.start_time`]

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``edata['rel_type']``: edge type
        """
        if t >= len(self) or t < 0:
            raise IndexError("Index out of range")
        i = t + self.start_time
        row_mask = self.time_index <= i
        edges = self.data[row_mask][:, [0, 2]]
        rate = self.data[row_mask][:, 1]
        g = dgl_graph((edges[:, 0], edges[:, 1]))
        g.edata["rel_type"] = F.tensor(
            rate.reshape(-1, 1), dtype=F.data_type_dict["int64"]
        )
        if self._transform is not None:
            g = self._transform(g)
        return g

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return self._end_time - self._start_time + 1

    @property
    def is_temporal(self):
        r"""Does the dataset contain temporal graphs

        Returns
        -------
        bool
        """
        return True


GDELT = GDELTDataset
