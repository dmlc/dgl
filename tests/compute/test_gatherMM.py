import time
import unittest
from timeit import default_timer

import backend as F
import numpy as np
import pytest
from test_utils import get_cases, parametrize_idtype

import dgl
import dgl.function as fn

iters = 5
n_edge_scale = 1
num_rel_scale = 1
