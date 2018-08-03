#!/bin/env python
from __future__ import print_function

from networkx import *
import cProfile

g = read_graphml("pgp.xml")

print("Profiling PageRank")
print("==================")
print()

cProfile.run("for i in range(10): pagerank_scipy(g, alpha=0.85, tol=1e-3, max_iter=10000000)", sort="cumulative")
