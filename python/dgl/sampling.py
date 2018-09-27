#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:43:03 2018

@author: ivabruge
"""

import collections as c
import numpy as np
import numpy.random as npr

#####
### Seed Methods
####




###
## Sample 'num_batches' lists of seeds (node IDs) each of length 'seed_size'
## @param V list of nodes
## @param seed_size int
## @param num_batches int
## @param @replace boolean
## @return a 2-D list of [[seeds], [seeds]....]
###
def seeds(V, seed_size, num_batches, replace=False):
    for i in range(num_batches):
        yield list(npr.choice(V, seed_size, replace=replace))
    return [list() for _ in range(num_batches)]

###
## Sample batches of size 'seed_size' until 'percent_nodes'*len(V) of 'V' is consumed. This will produce a variable number of seed-sets over a fixed number (but random selection) of nodes. To fix the source set, pre-sample and consume this list
## @param seed_size int
## @param num_nodes int
## @param percent_nodes float
## @param @replace boolean
## @return a 2-D list of [[seeds], [seeds]....]
###
def seeds_consume(V, seed_size, num_nodes=None, percent_nodes=.90):
    if num_nodes is None:
        num_nodes = int(np.round(len(V)*percent_nodes))
    node_list = npr.permutation(V)[0:min([num_nodes, len(V)])]
    l = len(node_list)
    for ndx in range(0, l, seed_size):
        r = list(node_list[ndx:min(ndx + seed_size, l)])
        if len(r) == seed_size:
            yield r

###
## Graph Traversal Methods
###



###
## This implements Importance Sampling (IS) node sampling in https://arxiv.org/abs/1801.10247
## This is merely sampling nodes porportional to degree. The authors then induce subgraphs over this set of sampled nodes 
## (This implements for a networkx object, implementation may vary slightly for other representations) 
## @param G a networkx graph 
## @param seed_size int the size of the node sample
## @param batches int the number of repeated node samples
## @param fn string the the sampling criteria to weight samples, (we're using degree, implement any others)
## @return list a list of batch dictionaries [d1, d2... di...] where dictionary d_i = {0: [nodes]} 
##     this representation allows for multiple levels in other sampling methods. This method has only one level.
## @return dict probability distribution where q[v] = degree(v)/|E| (for degree weighting)
###
def importance_sampling(q,seed_size=100, num_nodes=None,percent_nodes=.90):     
    if num_nodes is None:
        num_nodes = int(np.floor(len(q)*percent_nodes)) 
        for i in range(int(np.floor(num_nodes/seed_size))):
            r = {}
            r[0] = set([int(i) for i in npr.choice(list(q.keys()), size=seed_size, replace=False, p=list(q.values()))]) # select 'n' keys using probability distribution 'q.values()'  
            #ret.append(r)
            yield r

def importance_sampling_distribution_networkx(G, fn="degree"):
    q = {} #probability distribution for each q[v]
    if fn == "degree":
        degrees = G.degree
        s = sum(dict(degrees).values()) #total edges (directed)
        for v in G.nodes:
            q[v] = degrees[v]/s #my degree    
    return q
    
###
## Samples 'nodes' nodes based on hitting probability of a random walker over seed sets in list 'seedset_list' (advanced: using seed selection probability list 'weights' of len(seedset), assumes fixed-length seed-set)
## The walker randomly selects a node, walks with 'p_restart' probability of restarting, for 'iters' iterations. Each visitation is a hit. Nodes are ranked by hits selected in descending order.
##   
## Some walkers enforce no cycles. for speed, this one doesn't. 
## 
## This is the relevance sampling in the neighborhood sampling (NS) model: https://arxiv.org/abs/1806.01973 
##    
## @param G a graph G (in any representation we've implemented neighborhood)
## @param seedset_list A 2-D list of seeds (see 'Seed Methods' above)
## @param depth int the number of levels of nodes (e.g. in geodesic distance) to generate
## @param fn_neighborhood *function a function pointer implementing the neighborhood function on G (see 'Accessory Functions' below)
## @param max_level_nodes int, list, or None: the maximum number of nodes sampled per level. if a list, samples according to 'max_level_nodes[curr]' current node depth
## @param p_restart float the restart probability
## @param iters int the number of walking iterations
## @param weights a vector specifying weights for seed selection (default: integer, uniform sampling)
## @param max_path_length int the maximum path length, (so we can generate path lengths with a single bernoulli sampling)
## @return list a list of batch dictionaries [d1, d2... di...] where dictionary d_i = {0: [nodes], 1:[nodes]...} 
###
def seed_random_walk(G, seedset_list, depth, fn_neighborhood, max_level_nodes=None, p_restart=0.1, iters=1000, weights=1, max_path_length=20):
    
    if max_level_nodes is None:
        max_level_nodes = [np.inf for i in range(depth)]
    elif isinstance(max_level_nodes, int):
        max_level_nodes = [max_level_nodes for i in range(depth)]    
    
    #rr = []
    for S in seedset_list:
        dists = c.defaultdict(lambda:np.inf) #store best distances. Note: this approximates minimum path lengths, possible that dists[v] > minimum_path_length(v)
        hits = c.defaultdict(int) #store node hits
        if isinstance(weights, int): ##if unweighted
            s_iter = npr.choice(S, size=iters, replace=True)  #generate full list of seeds
        else:
            s_iter = npr.choice(S, size=iters, replace=True, p=weights)  #generate full list of seeds
        
        for s in s_iter: 
            l = list(npr.binomial(p_restart, 1, size=max_path_length)) + [1] #generate path length (ensure termination after 'size' steps)  
            for i in range(l.index(1)):
                hits[s]+= 1 #increment hits
                dists[s] = min(i, dists[s]) #check best distance
                neigh = fn_neighborhood(G, s)
                if not neigh: #no neighbors, end of path
                    break               
                s = int(npr.choice(neigh)) #choose single neighbor
        hits = {k:v for k,v in hits.items() if dists[k] <= depth-1} #filter hits by maximum depth

        d_sorted = sorted(hits, key=lambda k: hits[k])[::-1] ##sort filtered nodes by hits (descending)
        r = c.defaultdict(set)
        for key in d_sorted:
            k_iter = int(depth-1-dists[key])
            if len(r[k_iter]) < max_level_nodes[k_iter]: #enforce maximum number per level
                r[k_iter].add(int(key))
        yield(dict(r))
        #rr.append(dict(r))
    #return rr   

###
## Samples neighbors around a seed set in a BFS fashion, where we ensure exactly 'max_neighbors' of each sampled node are sampled
##
## @param G a graph G (in any representation we've implemented neighborhood)
## @param seedset_list A 2-D list of seeds (see 'Seed Methods' above) 
## @param depth int the number of levels of nodes (e.g. in geodesic distance) to generate
## @param fn_neighborhood *function a function pointer implementing the neighborhood function on G (see 'Accessory Functions' below)
## @param max_neighbors int, list, or None: the maximum number of neighbors sampled per visited node at the current depth. if a list, samples according to 'max_neighbors[curr]' current node depth
## @return list a list of batch dictionaries [d1, d2... di...] where dictionary d_i = {0: [nodes], 1:[nodes]...} 
##
###
def seed_neighbor_expansion(G,seedset_list,depth, fn_neighborhood, max_neighbors = None):
    
    if max_neighbors is None: #build max_neighbor list over depth
        max_neighbors = [np.inf for i in range(depth)]
    elif isinstance(max_neighbors, int):
        max_neighbors = [max_neighbors for i in range(depth)]
    
    #rr = []
    for S in seedset_list: #for seed-set
        d = {} #distances
        to_visit = set(S)  
        curr = 1
        
        for s in S: #init distance 0
            d[s] = 0
        
        while (len(to_visit)) and curr < depth: #while nodes to visit and more depth
            next_visit = set()       
            for s in to_visit: #for each node
                n_iter=0 #number of nodes added per 's'
                neigh = fn_neighborhood(G, s) 
                npr.shuffle(neigh) #randomize neighbors
                for v in neigh:               
                    if v not in d: #if v unseen
                        next_visit.add(v)
                        d[v] = curr #first we've seen this node, current depth
                        n_iter+=1 #increment number of nodes added for s
                        if n_iter >= max_neighbors[curr]: #if we've added neighbors for s
                            break                        
            curr+= 1
            to_visit = next_visit
        
        r = c.defaultdict(set)
        
        for key,d_k in d.items():
            r[int(depth-1-d_k)].add(int(key))
        yield(dict(r))
        #rr.append(dict(r))
    #return rr
    
## Samples neighbors around a seed set in a BFS fashion, where we ensure exactly 'max_level_nodes' are randomly sampled per level in depth
##
## @param G a graph G (in any representation we've implemented neighborhood)
## @param seedset_list A 2-D list of seeds (see 'Seed Methods' above) 
## @param depth int the number of levels of nodes (e.g. in geodesic distance) to generate
## @param fn_neighborhood *function a function pointer implementing the neighborhood function on G (see 'Accessory Functions' below)
## @param max_level_nodes int, list, or None: the maximum number of neighbors sampled per level over *all* visited nodes at this level. if a list, samples according to 'max_neighbors[curr]' current node depth
## @return list a list of batch dictionaries [d1, d2... di...] where dictionary d_i = {0: [nodes], 1:[nodes]...} 
    
def seed_BFS_frontier(G,seedset_list,depth, fn_neighborhood, max_level_nodes = None):
    
    if max_level_nodes is None:
        max_level_nodes = [np.inf for i in range(depth)]
    elif isinstance(max_level_nodes, int):
        max_level_nodes = [max_level_nodes for i in range(depth)]
    
    #rr = []
    for S in seedset_list: #for seedset
        d = {} #distances
        to_visit = set(S) 
        curr = 1
        
        for s in S: #initialize zero depth
            d[s] = 0
        
        while (len(to_visit)) and curr < depth: #while to nodes visit and more depth
            next_visit = set()       
            for s in to_visit: #for node to visit
                neigh = fn_neighborhood(G, s) #expand neighbors
                for v in neigh:
                    if v not in d: #if unseen
                        d[v] = curr 
                        next_visit.add(v) #add all unseen neighbors
            
            neigh = npr.permutation(list(next_visit)) #randomize unseen neighbors
            neigh = set(neigh[0:min(len(neigh), max_level_nodes[curr])]) #truncate to maximum number
            
            #kill unsampled neighbors with infinity
            if len(neigh):
                for v in next_visit:
                    if v not in neigh: #v not sampled
                        d[v] = np.inf
            to_visit = neigh #visit only sampled neighs
            curr+= 1
        r = c.defaultdict(set)
        
        for key,d_k in d.items():
            if d_k != np.inf: #if we didnt kill
                r[int(depth-1-d_k)].add(int(key))
        yield dict(r)
        #rr.append(dict(r))
    #return rr

###
# Seeded Graph Traversals
# Convenience methods combining seeding and minibatch sampling
### 
def seed_expansion_sampling(G, seed_size, depth, fn_neighborhood, max_neighbors = None, seed_nodes=None, percent_nodes=.90):
    seedset_iter = seeds_consume(G.nodes, seed_size =  seed_size, num_nodes=seed_nodes,percent_nodes=percent_nodes)
    return seed_neighbor_expansion(G,seedset_iter,depth, fn_neighborhood, max_neighbors=max_neighbors)

def seed_BFS_frontier_sampling(G,  seed_size, depth, fn_neighborhood, max_level_nodes = None, seed_nodes=None, percent_nodes=.90):
    seedset_iter = seeds_consume(G.nodes, seed_size =  seed_size, num_nodes=seed_nodes,percent_nodes=percent_nodes)
    return seed_BFS_frontier(G,seedset_iter,depth, fn_neighborhood, max_level_nodes=max_level_nodes)

def seed_random_walk_sampling(G, seed_size, depth, fn_neighborhood, max_level_nodes=None, seed_nodes=None, percent_nodes=.90, p_restart=0.3, iters=1000, weights=1, max_path_length=10):
    seedset_iter = seeds_consume(G.nodes, seed_size =  seed_size, num_nodes=seed_nodes,percent_nodes=percent_nodes)
    return seed_random_walk(G, seedset_iter, depth, fn_neighborhood, max_level_nodes=max_level_nodes, p_restart=p_restart, iters=iters, weights=weights, max_path_length=max_path_length)

####
## Accessories
#####

##http://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
#def chunk_iter(iterable, n=1):
#

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


####
####
######
## Minimal running example:
##


#import dgl.data as dgld
#import dgl.sampling as sampling
#import gcn.gcn_minibatch as gcn 
#import dgl
#
#params = gcn.default_params()
#data = dgld.load_data_dict(params)
#g = dgl.DGLGraph(data.graph)
#
#s_rw = sampling.seed_random_walk_sampling(g, params['fn_batch_params']["seed_size"], depth=params["layers"], fn_neighborhood=sampling.neighborhood_networkx, percent_nodes=params['fn_batch_params']["percent_nodes"])
#s_fr = s2 = sampling.seed_BFS_frontier_sampling(g, params['fn_batch_params']["seed_size"], depth=params["layers"], fn_neighborhood=sampling.neighborhood_networkx, percent_nodes=params['fn_batch_params']["percent_nodes"])
#s_se = sampling.seed_expansion_sampling(g, params['fn_batch_params']["seed_size"], depth=params["layers"], fn_neighborhood=sampling.neighborhood_networkx, percent_nodes=params['fn_batch_params']["percent_nodes"])
#s_is, q = sampling.importance_sampling_networkx(g)