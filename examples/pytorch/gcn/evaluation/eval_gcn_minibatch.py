#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:58:44 2018

@author: ivabruge
"""

import gcn.gcn_minibatch as gcn
import joblib as jl
import dgl.data as dgd
from dgl import DGLGraph
import dgl.sampling as sampling

import dgl.data as dgld
import dgl.sampling as sampling
import gcn.gcn_minibatch as gcn 
import dgl
import tqdm
import pickle


def batch_params(t=None):
    if t is None:
        return {'percent_nodes':percent_nodes,
                       'fn_neighborhood':sampling.neighborhood_networkx,
                       'depth':3}
    elif t == "is": 
        return {'percent_nodes':percent_nodes, "fn":"neighborhood"}
    
    
def generate_paramater_list(p_merge, d, seed_sizes,model_count=5, epochs=50,cv=None):    
    for k, e_iter in enumerate(v[v["expand"]]): 
        for i, s_iter in enumerate(seed_sizes):                  
                for j in range(model_count):
                    params_iter = {}
                    params_iter = gcn.default_params()
                    params_iter["fn_batch"] = v['fn']
                    if key == "IS":
                        params_iter["fn_batch_params"] = batch_params("is")
                    else:
                        params_iter["fn_batch_params"] = batch_params()
                        params_iter['fn_batch_params'][v["expand"]] = e_iter
                        
                    params_iter["gpu"]=-1
                    params_iter["cv"]=cv
                    params_iter["fn_batch_params"]["seed_size"] = s_iter
                    params_iter["epochs"]=epochs
                    params_iter["data"] = d 
                    yield ((key, e_iter,s_iter,j), params_iter)

def parallel_handler(key, value):
    print("[KEY]: "+str(key))
    return (key,gcn.main(value, data=value["data"]))

if __name__== '__main__':    
    seed_sizes = [100, 200, 300, 400]
    max_neighbors = [2,5, 7]
    max_level_nodes = [50, 100, 200]

    model_count = 10
    epochs = 20
    percent_nodes = 1
    cv = 6
    
    rets = {}
    datasets = ['cora'] 
    for d_iter in datasets:   
        
        params_iter = gcn.default_params()
        params_iter["dataset"] = d_iter
        d = dgld.load_data_dict(params_iter)

    
        p_merge = {'ES': {"expand": "max_neighbors", 'fn': sampling.seed_expansion_sampling,'max_neighbors':max_neighbors },
                      'RW': {"expand": "max_level_nodes", 'fn': sampling.seed_random_walk_sampling, 'max_level_nodes':max_level_nodes},
                      'FF': {"expand": "max_level_nodes",'fn': sampling.seed_BFS_frontier_sampling, 'max_level_nodes':max_level_nodes},
                      'IS': {"expand": "n",'fn':sampling.importance_sampling_wrapper_networkx, 'n':[None]}}
        ret ={}
        
        for key, v in p_merge.items():
            if key not in ret:
                params = generate_paramater_list(v,d, seed_sizes, model_count=model_count, epochs = epochs, cv=cv)
                #ret[key] = [parallel_handler(key, value) for key, value in params]
                ret[key] = jl.Parallel(n_jobs=4, verbose=10)(jl.delayed(parallel_handler)(a,b) for a,b in tqdm.tqdm(list(params)))      
        rets[d_iter] = ret
        if cv is not None:
            pickle.dump(ret, open('/Users/ivabruge/'+ str(d_iter) + '_cv.result', 'wb'))
        else:
            pickle.dump(ret, open('/Users/ivabruge/'+ str(d_iter) + '.result', 'wb'))