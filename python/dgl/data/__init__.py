"""The ``dgl.data`` package contains datasets hosted by DGL and also utilities
for downloading, processing, saving and loading data from external resources.
"""

from __future__ import absolute_import

from . import citation_graph as citegrh
from .actor import ActorDataset
from .movielens import MovieLensDataset
from .adapter import *
from .bitcoinotc import BitcoinOTC, BitcoinOTCDataset
from .citation_graph import (
    CitationGraphDataset,
    CiteseerGraphDataset,
    CoraBinary,
    CoraGraphDataset,
    PubmedGraphDataset,
)
from .csv_dataset import CSVDataset
from .dgl_dataset import DGLBuiltinDataset, DGLDataset
from .fakenews import FakeNewsDataset
from .flickr import FlickrDataset
from .fraud import FraudAmazonDataset, FraudDataset, FraudYelpDataset
from .gdelt import GDELT, GDELTDataset
from .gindt import GINDataset
from .gnn_benchmark import (
    AmazonCoBuy,
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    Coauthor,
    CoauthorCSDataset,
    CoauthorPhysicsDataset,
    CoraFull,
    CoraFullDataset,
)
from .icews18 import ICEWS18, ICEWS18Dataset
from .karate import KarateClub, KarateClubDataset
from .knowledge_graph import FB15k237Dataset, FB15kDataset, WN18Dataset
from .minigc import *
from .ppi import LegacyPPIDataset, PPIDataset
from .qm7b import QM7b, QM7bDataset
from .qm9 import QM9, QM9Dataset
from .qm9_edge import QM9Edge, QM9EdgeDataset
from .rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from .reddit import RedditDataset
from .sbm import SBMMixture, SBMMixtureDataset
from .synthetic import (
    BA2MotifDataset,
    BACommunityDataset,
    BAShapeDataset,
    TreeCycleDataset,
    TreeGridDataset,
)
from .tree import SST, SSTDataset
from .tu import LegacyTUDataset, TUDataset
from .utils import *
from .cluster import CLUSTERDataset
from .geom_gcn import (
    ChameleonDataset,
    CornellDataset,
    SquirrelDataset,
    TexasDataset,
    WisconsinDataset,
)

from .heterophilous_graphs import (
    AmazonRatingsDataset,
    MinesweeperDataset,
    QuestionsDataset,
    RomanEmpireDataset,
    TolokersDataset,
)

# RDKit is required for Peptides-Structural, Peptides-Functional dataset.
# Exception handling was added to prevent crashes for users who are using other
# datasets.
try:
    from .lrgb import (
        COCOSuperpixelsDataset,
        PeptidesFunctionalDataset,
        PeptidesStructuralDataset,
        VOCSuperpixelsDataset,
    )
except ImportError:
    pass
from .pattern import PATTERNDataset
from .superpixel import CIFAR10SuperPixelDataset, MNISTSuperPixelDataset
from .wikics import WikiCSDataset
from .yelp import YelpDataset
from .zinc import ZINCDataset


def register_data_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit",
    )


def load_data(args):
    if args.dataset == "cora":
        return citegrh.load_cora()
    elif args.dataset == "citeseer":
        return citegrh.load_citeseer()
    elif args.dataset == "pubmed":
        return citegrh.load_pubmed()
    elif args.dataset is not None and args.dataset.startswith("reddit"):
        return RedditDataset(self_loop=("self-loop" in args.dataset))
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
