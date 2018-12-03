"""Data related package."""
from __future__ import absolute_import

from . import citation_graph as citegrh
from .citation_graph import CoraBinary
from .tree import *
from .utils import *
from .sbm import SBMMixture

def register_data_args(parser):
    parser.add_argument("--dataset", type=str, required=False,
            help="The input dataset.")
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
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
