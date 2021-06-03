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
from dgl.sparse import libra2dgl_built_dict, libra2dgl_built_adj, libra2dgl_built_adj_v2, libra2dgl_fix_lf


def rep_per_node(prefix, nc):
    ifile = os.path.join(prefix, 'replicationlist.csv')
    f = open(ifile, "r")
    r_dt = {}

    fline = f.readline()
    for line in f:
        assert line[0] != '#', "Error read hash !!"
        node = line.strip('\n')
        if r_dt.get(node, -100) == -100:
            r_dt[node] = 1
        else:
            r_dt[node] += 1

    f.close()

    for v in r_dt.values():
        assert v < nc, "assertion in replication"

    return r_dt


def find_partition(nid, node_map):
    if nid == -1:
        return 1000
    
    pos = 0
    for nnodes in node_map:
        if nid < nnodes:
            return pos
        pos = pos +1
    print("Something is wrong in find_partition")
    sys.exit(0)


class args_:
    def __init__(self, dataset):
        self.dataset = dataset
        
    

## Code with C/C++ code extension
def main_ext(resultdir, dataset, nc):

    tedges = 1615685872
    hash_edges = [0, 0,
                  int((tedges/2)*1.2),
                  0,
                  int((tedges/4)*1.2),
                  0,0,0,
                  int((tedges/8)*1.2),
                  0, 0, 0, 0, 0, 0, 0,
                  int((tedges/16)*1.2),
                  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
                  int((tedges/32)*1.2),
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  int((tedges/64)*1.2),
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  int((tedges/128)*1.2)]
    
    ## load graph for the feature gather
    args = args_(dataset)

    print("Loading data", flush=True)                
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


    print("Done loading data", flush=True)    
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

        libra2dgl_built_dict(a, b, indices, ldt_key, gdt_key, gdt_value,
                             node_map, offset, nc, i, fsize, hash_nodes, resultdir)
        #libra2dgl_built_dict_v2(a, b, indices, ldt_key, gdt_key, gdt_value, lftensor,
        #                        node_map, offset, nc, i, fsize, hash_nodes)

        num_nodes = int(hash_nodes[0])
        num_edges = int(hash_nodes[1])
        part_nodes.append(num_nodes)
        
        #a_ = th.zeros(num_edges, dtype=th.int64)
        #b_ = th.zeros(num_edges, dtype=th.int64)        
            
        g.add_edges(a[0:num_edges], b[0:num_edges])

    ########################################################
    ## fixing lf at the split-nodes
    libra2dgl_fix_lf(gdt_key, gdt_value, lftensor, nc, N_n)
    ########################################################
    graph_name = dataset
    part_method = 'Libra'
    num_parts = nc
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
            labels = g_orig.ndata['label']
            one = True
        except:
            labels = g_orig.ndata['labels']
            
        #print("Labels: ", labels)
        trainm = g_orig.ndata['train_mask']
        testm = g_orig.ndata['test_mask']
        valm = g_orig.ndata['val_mask']
            
        feat_size = feat.shape[1]
        gfeat = th.zeros([num_nodes, feat_size], dtype=feat.dtype);

        #libra2dgl_built_adj(feat, gfeat, adj, inner_node, index, ldt, gdt_key,
        #                    gdt_value, node_map, num_nodes, nc, i, feat_size,
        #                    labels, trainm, testm, valm, glabels, gtrainm, gtestm, gvalm)
        libra2dgl_built_adj_v2(feat, gfeat, adj, inner_node, ldt, gdt_key,
                               gdt_value, node_map, lf, lftensor, num_nodes, nc, i, feat_size)
                                    

        g.ndata['adj'] = adj
        g.ndata['inner_node'] = inner_node
        g.ndata['feat'] = gfeat
        g.ndata['lf'] = lf

        ldt_ = []
        for l in range(num_nodes):
            if int(ldt[l]) >= g_orig.number_of_nodes() or int(ldt[l]) < 0:
                   print("ldt error: {} {}".format(int(ldt[l]), g_orig.number_of_nodes()) )
            assert int(ldt[l]) < g_orig.number_of_nodes(), "Error in ldt"
            assert int(ldt[l]) >= 0, "Error in ldt"
            ldt_.append(int(ldt[l]))
            
        ldt__ = th.tensor(ldt_)
        try:
            g.ndata['label'] = th.gather(g_orig.ndata['label'], 0, ldt__)
        except:
            g.ndata['label'] = th.gather(g_orig.ndata['labels'], 0, ldt__)
            
        g.ndata['train_mask'] = th.gather(g_orig.ndata['train_mask'], 0, ldt__)
        g.ndata['test_mask'] = th.gather(g_orig.ndata['test_mask'], 0, ldt__)
        g.ndata['val_mask'] = th.gather(g_orig.ndata['val_mask'], 0, ldt__)

        lf = g.ndata['lf']
        
        #print("Writing partition {} to file".format(i), flush=True)    
        
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
    part_dir = os.path.join("/cold_storage/omics/gnn/graph_partitioning/cagnet_datasets/" + dataset)
    #node_feats_file = os.path.join(part_dir + "/node_feat.dgl")
    graph_file = os.path.join(part_dir + "/graph.dgl")
    
    #node_feats = load_tensors(node_feats_file)
    graph = load_graphs(graph_file)[0][0]
    #graph.ndata = node_feats
    
    return graph


def main_ext2(resultdir, dataset, nc):

    tedges = 1615685872
    hash_edges = [0, 0,
                  int((tedges/2)*1.2),
                  0,
                  int((tedges/4)*1.2),
                  0,0,0,
                  int((tedges/8)*1.2),
                  0, 0, 0, 0, 0, 0, 0,
                  int((tedges/16)*1.2),
                  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
                  int((tedges/32)*1.2),
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  int((tedges/64)*1.2),
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  int((tedges/128)*1.2)]
    
    ## load graph for the feature gather
    args = args_(dataset)

    print("Loading data", flush=True)                
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
        try:
            g_orig = load_data(args)[0]
        except:
            print("Dataset {} not found !!!".format(dataset))
            sys.exit(1)


    print("Done loading data", flush=True)    
    print()    
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
        print("Community: ", i, flush=True)
        g = gg[i]
                
        fsize = hash_edges[nc]
                    
        hash_nodes = th.zeros(2, dtype=th.int32)
        a = th.zeros(fsize, dtype=th.int64)
        b = th.zeros(fsize, dtype=th.int64)
        ldt_key = th.zeros(fsize, dtype=th.int64)
        ldt_ar.append(ldt_key)

        libra2dgl_built_dict(a, b, indices, ldt_key, gdt_key, gdt_value,
                             node_map, offset, nc, i, fsize, hash_nodes, resultdir)
        #libra2dgl_built_dict_v2(a, b, indices, ldt_key, gdt_key, gdt_value, lftensor,
        #                        node_map, offset, nc, i, fsize, hash_nodes)

        num_nodes = int(hash_nodes[0])
        num_edges = int(hash_nodes[1])
        part_nodes.append(num_nodes)
                    
        g.add_edges(a[0:num_edges], b[0:num_edges])

    ########################################################
    ## fixing lf at the split-nodes
    libra2dgl_fix_lf(gdt_key, gdt_value, lftensor, nc, N_n)
    ########################################################
    graph_name = dataset
    part_method = 'Libra'
    num_parts = nc
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
            labels = g_orig.ndata['label']
            one = True
        except:
            labels = g_orig.ndata['labels']
            
        trainm = g_orig.ndata['train_mask']
        testm = g_orig.ndata['test_mask']
        valm = g_orig.ndata['val_mask']
            
        feat_size = feat.shape[1]
        gfeat = th.zeros([num_nodes, feat_size], dtype=feat.dtype);

        glabels = th.zeros(num_nodes, dtype=labels.dtype);
        gtrainm = th.zeros(num_nodes, dtype=trainm.dtype);
        gtestm = th.zeros(num_nodes, dtype=testm.dtype);
        gvalm = th.zeros(num_nodes, dtype=valm.dtype);
        
        libra2dgl_built_adj(feat, gfeat, adj, inner_node, ldt, gdt_key,
                            gdt_value, node_map, lf, lftensor, num_nodes, nc, i, feat_size,
                            labels, trainm, testm, valm, glabels, gtrainm, gtestm, gvalm,
                            feat.shape[0])
        #libra2dgl_built_adj_v2(feat, gfeat, adj, inner_node, ldt, gdt_key,
        #                       gdt_value, node_map, lf, lftensor, num_nodes, nc, i, feat_size)
                                    
        g.ndata['adj'] = adj
        g.ndata['inner_node'] = inner_node
        g.ndata['feat'] = gfeat
        g.ndata['lf'] = lf

        g.ndata['label'] = glabels
        g.ndata['train_mask'] = gtrainm
        g.ndata['test_mask'] = gtestm
        g.ndata['val_mask'] = gvalm

        for k in range(20):
            assert g.ndata['train_mask'][k] == gtrainm[k]
            assert g.ndata['test_mask'][k] == gtestm[k]
        

        lf = g.ndata['lf']
        #print("Writing partition {} to file".format(i), flush=True)    
        
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
    
    print("Completed !!!")

    return gg, node_map



def run(dataset, resultdir, nc, no_papers):
    th.set_printoptions(threshold=10)
        
    print("Dataset: ", dataset, flush=True)
    print("Result location: ",resultdir, flush=True)
    print("nc: ", nc)

    r_dt = rep_per_node(resultdir, nc)

    if no_papers:
        print("Executing small benchmark datasets")
        graph_ar, node_map =  main_ext(resultdir, dataset, nc)
    else:
        ## because labels is float in ogbn-papers
        print("Executing for specific case of ogbn-papers100M", flush=True)
        graph_ar, node_map =  main_ext2(resultdir, dataset, nc)      ##ogbn-papers special code
        
