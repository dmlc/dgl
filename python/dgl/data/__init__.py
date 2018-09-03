"""Data related package."""
from __future__ import absolute_import

from . import citation_graph as citegrh
from . import knowledge_graph as knwlgrh
from .tree import *
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
    elif args.dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(args)
    elif args.dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(args)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
