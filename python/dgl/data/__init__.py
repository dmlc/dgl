"""Data related package."""
from __future__ import absolute_import

from . import citation_graph as citegrh
from . import knowledge_graph as knwlgrh
from .utils import *

def register_data_args(parser):
    parser.add_argument("--dataset", type=str, required=True,
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
    elif args.dataset == 'aifb':
        return knwlgrh.load_aifb(args)
    elif args.dataset == 'mutag':
        return knwlgrh.load_mutag(args)
    elif args.dataset == 'bgs':
        return knwlgrh.load_bgs(args)
    elif args.dataset == 'am':
        return knwlgrh.load_am(args)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
