import networkx as nx
import pickle
import random

def convert_graph_to_ordering(g):
    ordering = []
    h = nx.DiGraph()
    h.add_edges_from(g.edges)
    for n in h.nodes():
        ordering.append(n)
        for m in h.predecessors(n):
            ordering.append((m, n))
    return ordering

def generate_dataset():
    n = 15
    m = 2
    n_samples = 1024
    samples = []
    for _ in range(n_samples):
        g = nx.barabasi_albert_graph(n, m)
        samples.append(convert_graph_to_ordering(g))

    with open('samples.p', 'wb') as f:
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
            self.ground_truth.append(
                    pad_ground_truth(datasets[i*batch_size: (i+1)*batch_size]))

    def __iter__(self):
        return iter(self.ground_truth)


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
    import torch
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
    while any([step[i] < length[i] for i in range(bz)]):
        node_select = []
        label_select = []
        mask = []
        for sample_idx in range(bz):
            if step[sample_idx] < length[sample_idx] and \
                    action[sample_idx][step[sample_idx]] == next_action:
                mask.append(1)
                node_select.append(node_list[sample_idx][step[sample_idx]])
                label_select.append(label[sample_idx][step[sample_idx]])
                step[sample_idx] += 1
            else:
                mask.append(0)
                node_select.append(-1)
                label_select.append(0)
        next_action = 1 - next_action
        new_node_list.append(torch.LongTensor(node_select))
        mask_for_batch.append(torch.ByteTensor(mask))
        new_label.append(torch.LongTensor(label_select))
        count += 1

    return count, new_label, new_node_list, mask_for_batch

if __name__ == '__main__':
    generate_dataset()
