from timeit import default_timer
import dgl
import backend as F
import dgl.function as fn
import time
import numpy as np
import unittest, pytest
from test_utils import parametrize_idtype, get_cases

iters = 5
n_edge_scale = 1
num_rel_scale = 1
