"""Data related package."""
from __future__ import absolute_import

from . import citation_graph as citegrh
from .citation_graph import CoraBinary
from .minigc import *
from .tree import *
from .utils import *
from .sbm import SBMMixture
from .reddit import RedditDataset
from .ppi import PPIDataset
from .tu import TUDataset
from .gindt import GINDataset


def register_data_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    citegrh.register_args(parser)


def load_data(args):
    if args.dataset == 'cora':
        return citegrh.load_cora()
    elif args.dataset == 'citeseer':
        return citegrh.load_citeseer()
    elif args.dataset == 'pubmed':
        return citegrh.load_pubmed()
    elif args.dataset == 'syn':
        return citegrh.load_synthetic(args)
    elif args.dataset is not None and args.dataset.startswith('reddit'):
        return RedditDataset(self_loop=('self-loop' in args.dataset))
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
