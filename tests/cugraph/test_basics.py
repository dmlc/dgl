import backend as F
import dgl
import numpy as np
from dgl import DGLGraph
import unittest
import pytest
import cugraph

def test_dummy():
    cg = cugraph.Graph()
    assert cg is not None
