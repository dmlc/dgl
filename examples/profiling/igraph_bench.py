#!/bin/env python

from igraph import *
import cProfile

g = Graph.Read_GraphML("pgp.xml")

print("Profiling shortest path")
print("=======================")
print()

cProfile.run("for i in range(1000): g.shortest_paths([g.vs[0]])", sort="cumulative")

print("Profiling PageRank")
print("==================")
print()

cProfile.run("for i in range(100): g.pagerank(damping=0.85)", sort="cumulative")

print("Profiling k-core")
print("================")
print()

cProfile.run("for i in range(1000): g.coreness(mode='all')", sort="cumulative")


print("Profiling minimum spanning tree")
print("===============================")
print()

cProfile.run("for i in range(1000): g.spanning_tree()", sort="cumulative")

'''
print("Profiling betweenness")
print("=====================")
print()

cProfile.run("for i in range(3): g.betweenness(); g.edge_betweenness()", sort="cumulative")
'''
