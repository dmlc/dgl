import dgl
import numpy as np
import torch as th


class Sampler:
    def __init__(
        self, graph, walk_length, num_walks, window_size, num_negative
    ):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negative = num_negative
        self.node_weights = self.compute_node_sample_weight()

    def sample(self, batch, sku_info):
        """
        Given a batch of target nodes, sample postive
        pairs and negative pairs from the graph
        """
        batch = np.repeat(batch, self.num_walks)

        pos_pairs = self.generate_pos_pairs(batch)
        neg_pairs = self.generate_neg_pairs(pos_pairs)

        # get sku info with id
        srcs, dsts, labels = [], [], []
        for pair in pos_pairs + neg_pairs:
            src, dst, label = pair
            src_info = sku_info[src]
            dst_info = sku_info[dst]

            srcs.append(src_info)
            dsts.append(dst_info)
            labels.append(label)

        return th.tensor(srcs), th.tensor(dsts), th.tensor(labels)

    def filter_padding(self, traces):
        for i in range(len(traces)):
            traces[i] = [x for x in traces[i] if x != -1]

    def generate_pos_pairs(self, nodes):
        """
        For seq [1, 2, 3, 4] and node NO.2,
        the window_size=1 will generate:
            (1, 2) and (2, 3)
        """
        # random walk
        traces, types = dgl.sampling.random_walk(
            g=self.graph, nodes=nodes, length=self.walk_length, prob="weight"
        )
        traces = traces.tolist()
        self.filter_padding(traces)

        # skip-gram
        pairs = []
        for trace in traces:
            for i in range(len(trace)):
                center = trace[i]
                left = max(0, i - self.window_size)
                right = min(len(trace), i + self.window_size + 1)
                pairs.extend([[center, x, 1] for x in trace[left:i]])
                pairs.extend([[center, x, 1] for x in trace[i + 1 : right]])

        return pairs

    def compute_node_sample_weight(self):
        """
        Using node degree as sample weight
        """
        return self.graph.in_degrees().float()

    def generate_neg_pairs(self, pos_pairs):
        """
        Sample based on node freq in traces, frequently shown
        nodes will have larger chance to be sampled as
        negative node.
        """
        # sample `self.num_negative` neg dst node
        # for each pos node pair's src node.
        negs = th.multinomial(
            self.node_weights,
            len(pos_pairs) * self.num_negative,
            replacement=True,
        ).tolist()

        tar = np.repeat([pair[0] for pair in pos_pairs], self.num_negative)
        assert len(tar) == len(negs)
        neg_pairs = [[x, y, 0] for x, y in zip(tar, negs)]

        return neg_pairs
