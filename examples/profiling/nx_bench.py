#!/bin/env python
from __future__ import print_function

from networkx import *
import cProfile

g = read_graphml("pgp.xml")

print("Profiling shortest path")
print("=======================")
print()

cProfile.run("for i in range(100): shortest_path_length(g, 'n0')", sort="cumulative")


print("Profiling PageRank")
print("==================")
print()

cProfile.run("for i in range(10): pagerank(g, alpha=0.85, tol=1e-3, max_iter=10000000)", sort="cumulative")

print("Profiling k-core")
print("================")
print()

cProfile.run("for i in range(10): core.core_number(g)", sort="cumulative")


print("Profiling minimum spanning tree")
print("===============================")
print()

u = g.to_undirected()

cProfile.run("for i in range(10): minimum_spanning_tree(u)", sort="cumulative")

'''
print("Profiling betweenness")
print("====================")
print()

cProfile.run("for i in range(1): betweenness_centrality(g); edge_betweenness_centrality(g)", sort="cumulative")
'''
