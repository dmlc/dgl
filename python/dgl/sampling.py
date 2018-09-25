#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:43:03 2018

@author: ivabruge
"""

import collections as c
import numpy as np
import numpy.random as npr


def seeds(V, seed_size, num_batches):
    return [list(npr.choice(V, seed_size, replace=False)) for _ in range(num_batches)]
    #return list(chunk_iter(npr.permutation(V), s))[0:k]

def seeds_consume(V, seed_size, num_nodes=None, percent_nodes=.90):
    if num_nodes is None:
        num_nodes = int(np.round(len(V)*percent_nodes))
    return list(chunk_iter(list(npr.permutation(V)[0:min([num_nodes, len(V)])]), seed_size))

#http://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def chunk_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def importance_sampling_networkx(G, n, levels, batches=100, fn="degree"):    
    q = {}
    ret = []
    if fn == "degree":
        degrees = G.degree
        s = sum(dict(degrees).values())
        for v in G.nodes:
            q[v] = degrees[v]/s
        for i in range(batches):
            r = {}
            for l in range(levels):
                r[l] = set([int(i) for i in npr.choice(list(q.keys()), size=n, replace=False, p=list(q.values()))])
            ret.append(r)
    return ret, q

def minibatch_seed_traverse(G,  seed_size, depth, fn_neighborhood, max_neighbors = np.inf, num_nodes=None, percent_nodes=.90):
    seedset_list = seeds_consume(G.nodes, seed_size =  seed_size, num_nodes=num_nodes,percent_nodes=percent_nodes)
    return minibatch_traverse(G,seedset_list,depth, fn_neighborhood, max_neighbors=max_neighbors)
    
def minibatch_traverse(G,seedset_list,depth, fn_neighborhood, max_neighbors = np.inf):
    rr = []
    for S in seedset_list:
        d = {}
        to_visit = set(S)  
        curr = 1
        
        for s in S:
            d[s] = 0
        
        while (len(to_visit)) and curr <= depth:
            next_visit = set()       
            for s in to_visit:
                if len(next_visit) >= max_neighbors:
                    break
                neigh = fn_neighborhood(G, s)
                npr.shuffle(neigh)
                #neigh = [int(v) for v in neigh]
                for v in neigh:
                    if v not in d:
                        d[v] = curr
                        next_visit.add(v)
                    if len(next_visit) >= max_neighbors:
                        break
            curr+= 1
            to_visit = next_visit
        
        r = c.defaultdict(set)
        
        #print(d)
        for key,d_k in d.items():
            r[int(depth-d_k)].add(int(key))
        rr.append(dict(r))
        #print(dict(r))
    return rr

def nodes_networkx(G):
    return G.nodes

def degree_networkx(G, i):
    return G

def neighborhood_networkx(G, i):
    return [n_i for n_i in G[i]]

def neighborhood_csr(G, i):
    return list(G[i].indices) 
    
def neighborhood_graphtool(G, i):
    return [int(s_i) for s_i in G.vertex(i).out_neighbors()]

def neighborhood_igraph(G, i):
    return G.neighbors(i, mode="OUT")    