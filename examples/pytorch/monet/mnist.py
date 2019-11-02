import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import GMMConv
from grid_graph import *
from coarsening import *


"""
class MoNet(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats):
"""

A = grid_graph(28, 8, 'euclidean')
print(type(A))

coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)


print(L, perm)