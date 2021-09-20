#/usr/bin/python

import os
import sys
import dgl
import json
from dgl import DGLGraph
import torch as th
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from load_graph import load_reddit, load_ogb
from dgl.sparse import libra2dgl_build_dict
from dgl.sparse import libra2dgl_set_lf
from dgl.sparse import libra2dgl_build_adjlist


## replication per node
def rep_per_node(prefix, nc):
    ifile = os.path.join(prefix, 'replicationlist.csv')
    f = open(ifile, "r")
    r_dt = {}

    fline = f.readline()
    for line in f:
        assert line[0] != '#', "Error: read hash !!"
        node = line.strip('\n')
        if r_dt.get(node, -100) == -100:
            r_dt[node] = 1
        else:
            r_dt[node] += 1

    f.close()

    for v in r_dt.values():
        assert v < nc, "Error: Assertion in replication !!"

    return r_dt


## Reports the partition for a given node
def find_partition(nid, node_map):
    if nid == -1:
        return 1000
    
    pos = 0
    for nnodes in node_map:
        if nid < nnodes:
            return pos
        pos = pos +1
    print("Error: Something is wrong in find_partition")
    sys.exit(1)


class Args:
    def __init__(self, dataset):
        self.dataset = dataset
        
    
## libra2dgl conversion C/C++ code extension
def main_libra2dgl(resultdir, dataset, nc):

    tedges = 1615685872   ## total edges
    max_c = 1024   ## max partitions supported    
    factor = 1.2
    
    ## for pre-allocated tensor size
    hash_edges_ = [0, 0,
                  int((tedges/2)*factor),
                  0,
                  int((tedges/4)*factor),
                  0,0,0,
                  int((tedges/8)*factor),
                  0, 0, 0, 0, 0, 0, 0,
                  int((tedges/16)*factor),
                  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
                  int((tedges/32)*factor),
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  int((tedges/64)*factor),
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  int((tedges/128)*factor)]

    hash_edges = [int((tedges/i)*factor) for i in range(1, max_c + 1)]
    
    ## load graph for the feature gather
    args = Args(dataset)

    print("Loading data...", flush=True)                
    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        g_orig, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        g_orig, _ = load_ogb('ogbn-papers100M')
    elif args.dataset == 'proteins':
        print("Loading proteins")
        g_orig = load_proteins('proteins')
    elif args.dataset == 'ogbn-arxiv':
        print("Loading ogbn-arxiv")
        g_orig, _ = load_ogb('ogbn-arxiv')                                        
    else:
        g_orig = load_data(args)[0]

    print("Done loading data.", flush=True)    
    a,b = g_orig.edges()

    N_n = g_orig.number_of_nodes()
    print("Number of nodes in the graph: ", N_n)
    node_map = th.zeros(nc, dtype=th.int32)
    indices = th.zeros(N_n, dtype=th.int32)
    lftensor = th.zeros(N_n, dtype=th.int32)
    gdt_key = th.zeros(N_n, dtype=th.int32)
    gdt_value = th.zeros([N_n, nc], dtype=th.int32)
    offset = th.zeros(1, dtype=th.int32)
    ldt_ar = []
    
    gg = [DGLGraph() for i in range(nc)]
    part_nodes = []
    
    for i in range(nc):
        g = gg[i]                
        fsize = hash_edges[nc]

        hash_nodes = th.zeros(2, dtype=th.int32)
        a = th.zeros(fsize, dtype=th.int64)
        b = th.zeros(fsize, dtype=th.int64)
        ldt_key = th.zeros(fsize, dtype=th.int64)
        ldt_ar.append(ldt_key)

        ## building node, parition dictionary
        libra2dgl_build_dict(a, b, indices, ldt_key, gdt_key, gdt_value,
                             node_map, offset, nc, i, fsize, hash_nodes,
                             resultdir)

        num_nodes = int(hash_nodes[0])
        num_edges = int(hash_nodes[1])
        part_nodes.append(num_nodes)
        
        g.add_edges(a[0:num_edges], b[0:num_edges])

    ########################################################
    ## fixing lf for the split-nodes
    libra2dgl_set_lf(gdt_key, gdt_value, lftensor, nc, N_n)
    ########################################################
    graph_name = dataset
    part_method = 'Libra'
    num_parts = nc   ## number of paritions/communities
    num_hops = 0
    node_map_val = node_map.tolist()
    edge_map_val = 0
    out_path = resultdir

    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g_orig.number_of_nodes(),
                     'num_edges': g_orig.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val}
    ############################################################
    
    for i in range(nc):
        g = gg[0]
        num_nodes = part_nodes[i]
        adj        = th.zeros([num_nodes, nc - 1], dtype=th.int32);
        inner_node = th.zeros(num_nodes, dtype=th.int32);
        lf         = th.zeros(num_nodes, dtype=th.int32);
        ldt = ldt_ar[0]

        try:
            feat = g_orig.ndata['feat']
        except:
            feat = g_orig.ndata['features']

        one = False
        try:
            #g_orig.ndata['label'] = g_orig.ndata['label'].type(th.float)
            labels = g_orig.ndata['label']
            one = True
        except:
            #g_orig.ndata['labels'] = g_orig.ndata['labels'].type(th.float)
            labels = g_orig.ndata['labels']
            
        #print("Labels: ", labels)
        trainm = g_orig.ndata['train_mask']
        testm = g_orig.ndata['test_mask']
        valm = g_orig.ndata['val_mask']
            
        feat_size = feat.shape[1]
        gfeat = th.zeros([num_nodes, feat_size], dtype=feat.dtype);

        glabels = th.zeros(num_nodes, dtype=labels.dtype);
        gtrainm = th.zeros(num_nodes, dtype=trainm.dtype);
        gtestm = th.zeros(num_nodes, dtype=testm.dtype);
        gvalm = th.zeros(num_nodes, dtype=valm.dtype);
        
        libra2dgl_build_adjlist(feat, gfeat, adj, inner_node, ldt, gdt_key,
                                gdt_value, node_map, lf, lftensor, num_nodes,
                                nc, i, feat_size, labels, trainm, testm, valm,
                                glabels, gtrainm, gtestm, gvalm, feat.shape[0])
                                    

        g.ndata['adj'] = adj
        g.ndata['inner_node'] = inner_node
        g.ndata['feat'] = gfeat
        g.ndata['lf'] = lf 

        g.ndata['label'] = glabels
        g.ndata['train_mask'] = gtrainm
        g.ndata['test_mask'] = gtestm
        g.ndata['val_mask'] = gvalm

        ## debug
        ##ldt_ = []
        ##for l in range(num_nodes):
        ##    if int(ldt[l]) >= g_orig.number_of_nodes() or int(ldt[l]) < 0:
        ##           print("ldt error: {} {}".format(int(ldt[l]), g_orig.number_of_nodes()) )
        ##    assert int(ldt[l]) < g_orig.number_of_nodes(), "Error in ldt"
        ##    assert int(ldt[l]) >= 0, "Error in ldt"
        ##    ldt_.append(int(ldt[l]))
        ##    
        ##ldt__ = th.tensor(ldt_)
        ##try:
        ##    #g.ndata['label'] = th.gather(g_orig.ndata['label'], 0, ldt__)
        ##    lab = th.gather(g_orig.ndata['label'], 0, ldt__)
        ##except:
        ##    #g.ndata['label'] = th.gather(g_orig.ndata['labels'], 0, ldt__)
        ##    lab = th.gather(g_orig.ndata['labels'], 0, ldt__)
        ##
        #### g.ndata['train_mask'] = th.gather(g_orig.ndata['train_mask'], 0, ldt__)
        #### g.ndata['test_mask'] = th.gather(g_orig.ndata['test_mask'], 0, ldt__)
        #### g.ndata['val_mask'] = th.gather(g_orig.ndata['val_mask'], 0, ldt__)
        ##tr = th.gather(g_orig.ndata['train_mask'], 0, ldt__)
        ##te = th.gather(g_orig.ndata['test_mask'], 0, ldt__)
        ##va = th.gather(g_orig.ndata['val_mask'], 0, ldt__)
        ##
        #### small test
        ##for k in range(g.ndata['label'].shape[0]):
        ##    assert g.ndata['train_mask'][k] == tr[k]
        ##    assert g.ndata['test_mask'][k] == te[k]
        ##    if not th.isnan(g.ndata['label'][k]):
        ##        assert g.ndata['label'][k] == lab[k]
        #print("DType: ", g.ndata['label'].dtype, " ", lab.dtype, " ", g.ndata['label'].shape[0])
        #print("Checks passed successfully!!")
        ## debug ends
            
        lf = g.ndata['lf']        
        print("Writing partition {} to file".format(i), flush=True)    
        
        part = g
        part_id = i
        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                    'edge_feats': edge_feat_file,
                                                    'part_graph': part_graph_file}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, part.ndata)
        save_graphs(part_graph_file, [part])
        
        del g
        del gg[0]
        del ldt
        del ldt_ar[0]
        
    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)
        

    return gg, node_map



def load_proteins(dataset):
    part_dir = dataset
    graph_file = os.path.join(part_dir + "/graph.dgl")
    graph = load_graphs(graph_file)[0][0]
    return graph
    



def run(dataset, resultdir, nc, no_papers):
    th.set_printoptions(threshold=10)
        
    print("Dataset: ", dataset, flush=True)
    print("Result location: ",resultdir, flush=True)
    print("number of parititons: ", nc)

    r_dt = rep_per_node(resultdir, nc)

    partition_ar, node_map =  main_libra2dgl(resultdir, dataset, nc)    
