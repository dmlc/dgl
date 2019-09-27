from __future__ import absolute_import

from . import knowledge_graph as knwlgrh
def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

from .aifb import load_aifb
from .bgs import load_bgs
from .am import load_am
from .mutag import load_mutag
from .rdf_graph import AIFB, MUTAG, BGS, AM
