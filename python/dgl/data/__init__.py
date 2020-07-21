"""Data related package."""
from __future__ import absolute_import

from . import citation_graph as citegrh
from . import synthetic_graph as syngrh
from .citation_graph import CoraBinary, CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from .minigc import *
from .tree import SST, SSTDataset
from .utils import *
from .sbm import SBMMixture, SBMMixtureDataset
from .reddit import RedditDataset
from .ppi import PPIDataset, LegacyPPIDataset
from .tu import TUDataset, LegacyTUDataset
from .gnn_benckmark import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset, \
    CoauthorCSDataset,CoraFullDataset, AmazonCoBuy, Coauthor, CoraFull
from .knowledge_graph import FB15k237Dataset, FB15kDataset, WN18Dataset
from .karate import KarateClubDataset, KarateClub
from .gindt import GINDataset
from .bitcoinotc import BitcoinOTC, BitcoinOTCDataset
from .gdelt import GDELT, GDELTDataset
from .icews18 import ICEWS18, ICEWS18Dataset
from .qm7b import QM7b, QM7bDataset
from .dgl_dataset import DGLDataset, DGLBuiltinDataset


def register_data_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    syngrh.register_args(parser)


def load_data(args):
    if args.dataset == 'cora':
        return citegrh.load_cora()
    elif args.dataset == 'citeseer':
        return citegrh.load_citeseer()
    elif args.dataset == 'pubmed':
        return citegrh.load_pubmed()
    elif args.dataset == 'syn':
        return syngrh.load_synthetic(args)
    elif args.dataset is not None and args.dataset.startswith('reddit'):
        return RedditDataset(self_loop=('self-loop' in args.dataset))
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
