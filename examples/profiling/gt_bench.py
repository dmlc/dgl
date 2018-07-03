#!/bin/env python

from graph_tool.all import *
import cProfile

g = load_graph("pgp.xml")

print("Profiling shortest path")
print("=======================")
print()

cProfile.run("for i in range(1000): shortest_distance(g, g.vertex(0))", sort="cumulative")

print("Profiling PageRank")
print("==================")
print()

cProfile.run("for i in range(100): pagerank(g, damping=0.85, epsilon=1e-3)", sort="cumulative")

print("Profiling k-core")
print("================")
print()

cProfile.run("for i in range(1000): kcore_decomposition(g)", sort="cumulative")
# cProfile.run("for i in range(1000): kcore_decomposition(g, deg='total')", sort="cumulative")

print("Profiling minimum spanning tree")
print("===============================")
print()

cProfile.run("for i in range(1000): min_spanning_tree(g)", sort="cumulative")

'''
print("Profiling betweenness")
print("=====================")
print()

cProfile.run("for i in range(3): betweenness(g)", sort="cumulative")
'''
