#-*- coding:utf-8 -*-


import torch as th
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader

def build_dummy_comp_data():
    """
    This dummy data simulate the graph in CompGCN paper figure 1. Here there are 4 types of nodes:
    1. User, e.g. Christopher Nolan
    2. City, e.g. City of London
    3. Country, e.g. United Kindom
    4. Film, e.g. Dark Knight
    The figure 1 is very simple, one node of each type, and only contains 3 relations among them
    1. Born_in, e.g. Nolan was born_in City of London
    2. Citizen_of, e.g. Nolan is citizen_of United Kingdom
    3. Directed_by, e.g. Film Dark Knight is directed_by Nolan

    Returns
    -------
    A DGLGraph with 5 nodes of 4 types, and 3 edges in 3 types.

    """
    g = dgl.heterograph(
        {
         ('user', 'born_in', 'city'): ([th.tensor(0)], [th.tensor(0)]),
         ('user', 'citizen_of', 'country'): ([th.tensor(0)], [th.tensor(0)]),
         ('film', 'directed_by', 'user'): ([th.tensor(0), th.tensor(1)], [th.tensor(0),th.tensor(0)]),
         # add inversed edges
         ('city', 'born_in_inv', 'user'): ([th.tensor(0)], [th.tensor(0)]),
         ('country', 'citizen_of_inv', 'user'): ([th.tensor(0)], [th.tensor(0)]),
         ('user', 'directed_by_inv', 'film'): ([th.tensor(0), th.tensor(0)], [th.tensor(0), th.tensor(1)])
        }
    )

    n_feats = {
        'user': th.ones(1, 5),
        'city': th.ones(1, 5) * 2,
        'country': th.ones(1, 5) * 4,
        'film': th.ones(2, 5) * 8
    }

    e_feats = {
        'born_in': th.ones(1, 5) * 0.5,
        'citizen_of': th.ones(1, 5) * 0.5 * 0.5,
        'directed_by': th.ones(1, 5) * 0.5 * 0.5 * 0.5,
        'born_in_inv': th.ones(1, 5) * 0.5,
        'citizen_of_inv': th.ones(1, 5) * 0.5 * 0.5,
        'directed_by_inv': th.ones(1, 5) * 0.5 * 0.5 * 0.5
    }

    return g, n_feats, e_feats

