"""
Wikipedia page-page networks on three topics: chameleons, crocodiles and
squirrels
"""
from .dgl_dataset import DGLBuiltinDataset


class WikiNetworkDataset(DGLBuiltinDataset):
    r"""Wikipedia page-page networks from `Multi-scale Attributed Node
    Embedding <https://arxiv.org/abs/1909.13021>`__

    Parameters
    ----------
    name : str
        Name of the dataset.
    raw_dir : str
        Raw file directory to store the processed data.
    force_reload : bool
        Whether to always generate the data from scratch rather than load a
        cached version.
    verbose : bool
        Whether to print progress information.
    transform : callable
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """
    def __init__(
        self,
        name,
        raw_dir,
        force_reload,
        verbose,
        transform
    ):
        # TODO: provide url
        super(WikiNetworkDataset, self).__init__(
            name=name,
            url=None,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform
        )

    def process(self):
        """Load and process the data."""


class ChameleonDataset(WikiNetworkDataset):
    """Wikipedia page-page network on chameleons from `Multi-scale Attributed
    Node Embedding <https://arxiv.org/abs/1909.13021>`__

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to store the processed data. Default: ~/.dgl/
    force_reload : bool, optional
        Whether to always generate the data from scratch rather than load a
        cached version. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access. Default: None

    Examples
    --------
    """
    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None
    ):
        super(ChameleonDataset, self).__init__(
            name='chameleon',
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform
        )
