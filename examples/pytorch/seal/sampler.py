import torch
import dgl
from torch.utils.data import DataLoader, Dataset
from dgl import DGLGraph, NID, EID
from utils import drnl_node_labeling, coalesce_graph
from dgl.dataloading.negative_sampler import Uniform
from dgl import add_self_loop
import os.path as osp
from tqdm import tqdm
from copy import deepcopy


class GraphDataSet(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, graph_list, tensor):
        self.graph_list = graph_list
        self.tensor = tensor

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.tensor[index])


class SEALDataLoader(object):
    """
    Data Loader of SEAL
    Attributes:
        dataset(Dataset): dataset
        batch_size(int): size of batch
    """

    def __init__(self, dataset, batch_size, num_workers=1, shuffle=True,
                 drop_last=False, pin_memory=False):
        self.total_graphs = len(dataset)
        self.dataloader = DataLoader(dataset=dataset, collate_fn=self._collate, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)

    def _collate(self, batch):
        batch_graphs, batch_labels = map(list, zip(*batch))

        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        return batch_graphs, batch_labels

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)


class PosNegEdgesGenerator(object):
    """
    Generate positive and negative samples
    Attributes:
        g(dgl.DGLGraph): graph
        split_edge(dict): split edge
        neg_samples(int): num of negative samples per positive sample
        subsample_ratio(float): ratio of subsample
        shuffle(bool): if shuffle generated graph list
    """

    def __init__(self, g, split_edge, neg_samples=1, subsample_ratio=0.1, shuffle=True, return_type='combine'):
        self.neg_sampler = Uniform(neg_samples)
        self.subsample_ratio = subsample_ratio
        self.split_edge = split_edge
        self.g = g
        self.return_type = return_type
        self.shuffle = shuffle

    def __call__(self, split_type):

        if split_type == 'train':
            subsample_ratio = self.subsample_ratio
        else:
            subsample_ratio = 1

        pos_edges = self.split_edge[split_type]['edge']
        if split_type == 'train':
            g = add_self_loop(self.g)
            eids = g.edge_ids(pos_edges[:, 0], pos_edges[:, 1])
            neg_edges = torch.stack(self.neg_sampler(g, eids), dim=1)
        else:
            neg_edges = self.split_edge[split_type]['edge_neg']
        pos_edges = self.subsample(pos_edges, subsample_ratio).long()
        neg_edges = self.subsample(neg_edges, subsample_ratio).long()

        if self.return_type == 'split':
            return pos_edges, torch.ones(pos_edges.size(0)), neg_edges, torch.zeros(neg_edges.size(0))
        elif self.return_type == 'combine':
            edges = torch.cat([pos_edges, neg_edges])
            labels = torch.cat([torch.ones(pos_edges.size(0), 1), torch.zeros(neg_edges.size(0), 1)])
            if self.shuffle:
                perm = torch.randperm(edges.size(0))
                edges = edges[perm]
                labels = labels[perm]
            return edges, labels

    def subsample(self, edges, subsample_ratio):

        num_edges = edges.size(0)
        perm = torch.randperm(num_edges)
        perm = perm[:int(subsample_ratio * num_edges)]
        edges = edges[perm]
        return edges


class EdgeDataSet(Dataset):
    """
    Assistant Dataset for speeding up the SEALSampler
    """

    def __init__(self, edges, labels, transform):
        self.edges = edges
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        subgraph = self.transform(self.edges[index])
        return (subgraph, self.labels[index])


class SEALSampler(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        num_workers(int): num of workers

    """

    def __init__(self, graph, hop=1, num_workers=32, print_fn=print):
        self.graph = graph
        self.hop = hop
        self.print_fn = print_fn
        self.num_workers = num_workers

    def sample_subgraph(self, target_nodes):
        """
        Args:
            target_nodes(Tensor): Tensor of two target nodes
        Returns:
            subgraph(DGLGraph): subgraph
        """
        sample_nodes = [target_nodes]
        frontiers = target_nodes

        for i in range(self.hop):
            frontiers = self.graph.out_edges(frontiers)[1]
            frontiers = torch.unique(frontiers)
            sample_nodes.append(frontiers)

        sample_nodes = torch.cat(sample_nodes)
        sample_nodes = torch.unique(sample_nodes)
        subgraph = dgl.node_subgraph(self.graph, sample_nodes)

        # Each node should have unique node id in the new subgraph
        u_id = int(torch.nonzero(subgraph.ndata[NID] == int(target_nodes[0]), as_tuple=False))
        v_id = int(torch.nonzero(subgraph.ndata[NID] == int(target_nodes[1]), as_tuple=False))
        # remove link between target nodes in positive subgraphs.
        # Edge removing will rearange NID and EID, which lose the original NID and EID.
        if dgl.__version__[:5] < '0.6.0':
            nids = subgraph.ndata[NID]
            eids = subgraph.edata[EID]
            if subgraph.has_edges_between(u_id, v_id):
                link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
                subgraph.remove_edges(link_id)
                eids = eids[subgraph.edata[EID]]
            if subgraph.has_edges_between(v_id, u_id):
                link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
                subgraph.remove_edges(link_id)
                eids = eids[subgraph.edata[EID]]
            subgraph.ndata[NID] = nids
            subgraph.edata[EID] = eids
        else:
            if subgraph.has_edges_between(u_id, v_id):
                link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
                subgraph.remove_edges(link_id)
            if subgraph.has_edges_between(v_id, u_id):
                link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
                subgraph.remove_edges(link_id)

        z = drnl_node_labeling(subgraph, u_id, v_id)
        subgraph.ndata['z'] = z

        return subgraph

    def _collate(self, batch):

        batch_graphs, batch_labels = map(list, zip(*batch))

        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        return batch_graphs, batch_labels

    def __call__(self, edges, labels):
        subgraph_list = []
        labels_list = []
        edge_dataset = EdgeDataSet(edges, labels, transform=self.sample_subgraph)
        self.print_fn('Using {} workers in sampling job.'.format(self.num_workers))
        sampler = DataLoader(edge_dataset, batch_size=32, num_workers=self.num_workers,
                             shuffle=False, collate_fn=self._collate)
        for subgraph, label in tqdm(sampler, ncols=100):
            label_copy = deepcopy(label)
            subgraph = dgl.unbatch(subgraph)

            del label
            subgraph_list += subgraph
            labels_list.append(label_copy)

        return subgraph_list, torch.cat(labels_list)


class SEALData(object):
    """
    1. Generate positive and negative samples
    2. Subgraph sampling

    Attributes:
        g(dgl.DGLGraph): graph
        split_edge(dict): split edge
        hop(int): num of hop
        neg_samples(int): num of negative samples per positive sample
        subsample_ratio(float): ratio of subsample
        use_coalesce(bool): if coalesce graph.
    """

    def __init__(self, g, split_edge, hop=1, neg_samples=1, subsample_ratio=1, prefix=None, save_dir=None,
                 num_workers=32, shuffle=True, use_coalesce=True, print_fn=print):
        self.g = g
        self.hop = hop
        self.subsample_ratio = subsample_ratio
        self.prefix = prefix
        self.save_dir = save_dir
        self.print_fn = print_fn

        self.generator = PosNegEdgesGenerator(g=self.g,
                                              split_edge=split_edge,
                                              neg_samples=neg_samples,
                                              subsample_ratio=subsample_ratio,
                                              shuffle=shuffle,
                                              return_type='combine')
        if use_coalesce:
            self.g = coalesce_graph(self.g, copy_data=True)

        self.ndata = {k: v for k, v in self.g.ndata.items()}
        self.edata = {k: v for k, v in self.g.edata.items()}
        self.g.ndata.clear()
        self.g.edata.clear()
        self.print_fn("Save ndata and edata in class.")
        self.print_fn("Clear ndata and edata in graph.")

        self.sampler = SEALSampler(graph=self.g,
                                   hop=hop,
                                   num_workers=num_workers,
                                   print_fn=print_fn)

    def __call__(self, split_type):

        if split_type == 'train':
            subsample_ratio = self.subsample_ratio
        else:
            subsample_ratio = 1

        path = osp.join(self.save_dir or '', '{}_{}_{}-hop_{}-subsample.bin'.format(self.prefix, split_type,
                                                                                    self.hop, subsample_ratio))

        if osp.exists(path):
            self.print_fn("Load existing processed {} files".format(split_type))
            graph_list, data = dgl.load_graphs(path)
            dataset = GraphDataSet(graph_list, data['labels'])

        else:
            self.print_fn("Processed {} files not exist.".format(split_type))

            edges, labels = self.generator(split_type)
            self.print_fn("Generate {} edges totally.".format(edges.size(0)))

            graph_list, labels = self.sampler(edges, labels)
            dataset = GraphDataSet(graph_list, labels)
            dgl.save_graphs(path, graph_list, {'labels': labels})
            self.print_fn("Save preprocessed subgraph to {}".format(path))
        return dataset
