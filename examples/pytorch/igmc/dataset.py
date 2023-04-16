import torch as th
import dgl

from data import MovieLens
from torch.utils.data import Dataset
from utils import igmc_subgraph_extraction_labeling, one_hot

class IGMCMovieLens(Dataset):
    def __init__(self, movielens,
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200, mode='train'):
        super(IGMCMovieLens, self).__init__()    
        assert mode in ['train', 'test'], f"unrecorgnized mode {mode}"
        self.movielens = movielens
        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

        self.graph = self.movielens.train_graph if mode == 'train' else self.movielens.test_graph
        self.rating_pairs = self.movielens.info['train_rating_pairs'] if mode == 'train' else self.movielens.info['test_rating_pairs']

    def __len__(self):
        return int(self.graph.num_edges() / 2)

    def __getitem__(self, idx):
        u, v = self.rating_pairs[0][idx], self.rating_pairs[1][idx]
        g_label = self.graph.edata['etype'][idx]

        subgraph = igmc_subgraph_extraction_labeling(
            self.graph, (u, v), h=self.hop, 
            sample_ratio=self.sample_ratio, max_nodes_per_hop=self.max_nodes_per_hop)
        
        subgraph.ndata['nlabel'] = one_hot(subgraph.ndata['nlabel'], (self.hop+1)*2)
        
        # set edge mask to zero as to remove edges between target nodes in training process
        subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges())
        su = subgraph.nodes()[subgraph.ndata[dgl.NID]==u]
        sv = subgraph.nodes()[subgraph.ndata[dgl.NID]==v]
        _, _, target_edges = subgraph.edge_ids([su, sv], [sv, su], return_uv=True)
        subgraph.edata['edge_mask'][target_edges] = 0
        
        return subgraph, g_label

def collate_movielens(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label

if __name__ == '__main__':
    movielens = MovieLens(force_reload=True)
    data = IGMCMovieLens(movielens)
    g = data[0]
    pass