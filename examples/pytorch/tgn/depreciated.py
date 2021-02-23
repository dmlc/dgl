class FastTemporalLoader:
    def __init__(self,g,batch_size,n_neighbors,sampling_method='topk'):
        self.g  = g
        num_edges = g.num_edges()
        num_nodes = g.num_nodes()
        self.div_dict = {'train':int(TRAIN_SPLIT*num_edges),
                    'valid':int(VALID_SPLIT*num_edges),
                    'test':num_edges}
        self.dynamic_graph = dgl.graph(([],[]))
        self.batch_size = batch_size
        self.batch_cnt = 0
        self.nodes = set()
        self.dsts = set()
        if sampling_method == 'topk':
            self.sampler = partial(dgl.sampling.select_topk,k=n_neighbors,weight='timestamp')
        else:
            self.sampler = partial(dgl.sampling.sample_neighbors,fanout=n_neighbors)
    
    # Should be the same
    def get_next_batch(self,mode='train'):
        
        done = False
        working_div = self.div_dict[mode]
        src_list = self.g.edges()[0][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,working_div)]
        dst_list = self.g.edges()[1][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,working_div)]
        t_stamps = self.g.edata['timestamp'][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,working_div)]
        feats = self.g.edata['feats'][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,working_div)]
        edge_ids = range(self.batch_cnt*self.batch_size,min((self.batch_cnt+1)*self.batch_size,working_div))
        subgraph = dgl.edge_subgraph(self.g,edge_ids)
        self.batch_cnt += 1
        if subgraph.num_edges() < self.batch_size:
            done = True
        return done, src_list,dst_list,t_stamps,feats,subgraph

    def add_edges(self,srcs,dsts,timestamp,feats):
        self.dynamic_graph.add_edges(srcs,dsts,
                                    {'timestamp':timestamp,'feats':feats})
        self.nodes = self.nodes.union(set(srcs.tolist()+dsts.tolist()))
        self.dsts  = self.dsts.union(set(dsts.tolist()))
        #print("Set:",self.nodes)

    # Temporal Sampling Module
    def get_nodes_affiliation(self,nodes):
        if type(nodes) != list:
            nodes = [int(nodes)]
        else:
            nodes = [int(nodes[i]) for i in range(len(nodes))]
        
        graph = dgl.add_reverse_edges(self.dynamic_graph,copy_edata=True)
        nodes_set = set(nodes)
        frontier = list(nodes_set.intersection(self.nodes))
        new_node = list(nodes_set-self.nodes)
        
        full_neighbor_subgraph = dgl.in_subgraph(graph,frontier)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,new_node,new_node)
        
        # Remove orphan nodes
        non_orphan = (full_neighbor_subgraph.in_degrees()>0)+(full_neighbor_subgraph.out_degrees()>0)
        final_subgraph = dgl.node_subgraph(full_neighbor_subgraph,non_orphan)
        root2sub_dict = dict(zip(final_subgraph.ndata[dgl.NID].tolist(),final_subgraph.nodes().tolist()))
        nodes = [root2sub_dict[n] for n in nodes]
        
        final_subgraph = self.sampler(g=final_subgraph,nodes=nodes)
        
        final_subgraph = dgl.remove_self_loop(final_subgraph)
        final_subgraph = dgl.add_self_loop(final_subgraph)
        

        return nodes,final_subgraph
    
    def reset(self):
        self.batch_cnt = 0

