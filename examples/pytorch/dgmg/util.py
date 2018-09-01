import networkx as nx
import pickle
import random
import dgl
import numpy as np
import torch

def convert_graph_to_ordering(g):
    ordering = []
    h = nx.DiGraph()
    h.add_edges_from(g.edges)
    for n in range(len(h)):
        ordering.append(n)
        for m in h.predecessors(n):
            ordering.append((m, n))
    return ordering

def generate_dataset(n, m, n_samples, fname):
    samples = []
    for _ in range(n_samples):
        g = nx.barabasi_albert_graph(n, m)
        samples.append(convert_graph_to_ordering(g))

    with open(fname, 'wb') as f:
        pickle.dump(samples, f)

class DataLoader(object):
    def __init__(self, fname, batch_size, shuffle=True):
        with open(fname, 'rb') as f:
            datasets = pickle.load(f)
        if shuffle:
            random.shuffle(datasets)
        num = len(datasets) // batch_size

        # pre-process dataset
        self.ground_truth = []
        for i in range(num):
            batch = datasets[i*batch_size: (i+1)*batch_size]
            padded_signals = pad_ground_truth(batch)
            merged_graph = generate_merged_graph(batch)
            self.ground_truth.append([padded_signals, merged_graph])

    def __iter__(self):
        return iter(self.ground_truth)


def generate_merged_graph(batch):
    n_graphs = len(batch)
    graph_list = []
    # build each sample graph
    new_edges = []
    for ordering in batch:
        g = dgl.DGLGraph()
        node_count = 0
        edge_list = []
        for step in ordering:
            if isinstance(step, int):
                node_count += 1
            else:
                assert isinstance(step, tuple)
                edge_list.append(step)
                edge_list.append(tuple(reversed(step)))
        g.add_nodes_from(range(node_count))
        g.add_edges_from(edge_list)
        new_edges.append(zip(*edge_list))
        graph_list.append(g)
    # batch
    bg = dgl.batch(graph_list)
    # get new edges
    new_edges = [bg.query_new_edge(g, *edges) for g, edges in zip(graph_list, new_edges)]
    new_src, new_dst = zip(*new_edges)
    return bg, new_src, new_dst

def expand_ground_truth(ordering):
    node_list = []
    action = []
    label = []
    first_step = True
    for i in ordering:
        if isinstance(i, int):
            if not first_step:
                # add not to add edge
                action.append(1)
                label.append(0)
                node_list.append(-1)
            else:
                first_step = False
            action.append(0) # add node
            label.append(1)
            node_list.append(i)
        else:
            assert(isinstance(i, tuple))
            action.append(1)
            label.append(1)
            node_list.append(i[0]) # select src node to add
    # add not to add node
    action.append(0)
    label.append(0)
    node_list.append(-1)
    return len(action), action, label, node_list

def pad_ground_truth(batch):
    a = []
    bz = len(batch)
    for sample in batch:
        a.append(expand_ground_truth(sample))
    length, action, label, node_list = zip(*a)
    step = [0] * bz
    new_label = []
    new_node_list = []
    mask_for_batch = []
    next_action = 0
    count = 0
    active_step = [] # steps at least some graphs are not masked
    label1_set = [] # graphs who decide to add node or edge
    label1_set_tensor = []
    while any([step[i] < length[i] for i in range(bz)]):
        node_select = []
        label_select = []
        mask = []
        label1 = []
        not_all_masked = False
        for sample_idx in range(bz):
            if step[sample_idx] < length[sample_idx] and \
                    action[sample_idx][step[sample_idx]] == next_action:
                mask.append(1)
                node_select.append(node_list[sample_idx][step[sample_idx]])
                label_select.append(label[sample_idx][step[sample_idx]])
                # if decide to add node or add edge, record sample_idx
                if label_select[-1] == 1:
                    label1.append(sample_idx)
                step[sample_idx] += 1
                not_all_masked = True
            else:
                mask.append(0)
                node_select.append(-1)
                label_select.append(0)
        next_action = 1 - next_action
        new_node_list.append(torch.LongTensor(node_select))
        mask_for_batch.append(torch.ByteTensor(mask))
        new_label.append(torch.LongTensor(label_select))
        active_step.append(not_all_masked)
        label1_set.append(np.array(label1))
        label1_set_tensor.append(torch.LongTensor(label1))
        count += 1

    return count, new_label, new_node_list, mask_for_batch, active_step, label1_set, label1_set_tensor

def elapsed(msg, start, end):
    print("{}: {} ms".format(msg, int((end-start)*1000)))

if __name__ == '__main__':
    n = 15
    m = 2
    n_samples = 1024
    fname ='samples.p'
    generate_dataset(n, m, n_samples, fname)
