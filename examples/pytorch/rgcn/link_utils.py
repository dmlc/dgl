"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch as th
import dgl

# Utility function for building training and testing graphs

def get_subset_g(g, mask, num_rels, bidirected=False):
    src, dst = g.edges()
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata['etype'][mask]

    if bidirected:
        sub_src, sub_dst = th.cat([sub_src, sub_dst]), th.cat([sub_dst, sub_src])
        sub_rel = th.cat([sub_rel, sub_rel + num_rels])

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata[dgl.ETYPE] = sub_rel

    return sub_g

def preprocess(g, num_rels):
    # Get train graph
    train_g = get_subset_g(g, g.edata['train_mask'], num_rels)

    # Get test graph
    test_g = get_subset_g(g, g.edata['train_mask'], num_rels, bidirected=True)
    test_g.edata['norm'] = dgl.norm_by_dst(test_g).unsqueeze(-1)

    return train_g, test_g

class GlobalUniform:
    def __init__(self, g, sample_size):
        self.sample_size = sample_size
        self.eids = np.arange(g.num_edges())

    def sample(self):
        return th.from_numpy(np.random.choice(self.eids, self.sample_size))

class NeighborExpand:
    """Sample a connected component by neighborhood expansion"""
    def __init__(self, g, sample_size):
        self.g = g
        self.nids = np.arange(g.num_nodes())
        self.sample_size = sample_size

    def sample(self):
        edges = th.zeros((self.sample_size), dtype=th.int64)
        neighbor_counts = (self.g.in_degrees() + self.g.out_degrees()).numpy()
        seen_edge = np.array([False] * self.g.num_edges())
        seen_node = np.array([False] * self.g.num_nodes())

        for i in range(self.sample_size):
            if np.sum(seen_node) == 0:
                node_weights = np.ones_like(neighbor_counts)
                node_weights[np.where(neighbor_counts == 0)] = 0
            else:
                # Sample a visited node if applicable.
                # This guarantees a connected component.
                node_weights = neighbor_counts * seen_node

            node_probs = node_weights / np.sum(node_weights)
            chosen_node = np.random.choice(self.nids, p=node_probs)

            # Sample a neighbor of the sampled node
            u1, v1, eid1 = self.g.in_edges(chosen_node, form='all')
            u2, v2, eid2 = self.g.out_edges(chosen_node, form='all')
            u = th.cat([u1, u2])
            v = th.cat([v1, v2])
            eid = th.cat([eid1, eid2])

            to_pick = True
            while to_pick:
                random_id = th.randint(high=eid.shape[0], size=(1,))
                chosen_eid = eid[random_id]
                to_pick = seen_edge[chosen_eid]

            chosen_u = u[random_id]
            chosen_v = v[random_id]
            edges[i] = chosen_eid
            seen_node[chosen_u] = True
            seen_node[chosen_v] = True
            seen_edge[chosen_eid] = True

            neighbor_counts[chosen_u] -= 1
            neighbor_counts[chosen_v] -= 1

        return edges

class NegativeSampler:
    def __init__(self, k=10):
        self.k = k

    def sample(self, pos_samples, num_nodes):
        batch_size = len(pos_samples)
        neg_batch_size = batch_size * self.k
        neg_samples = np.tile(pos_samples, (self.k, 1))

        values = np.random.randint(num_nodes, size=neg_batch_size)
        choices = np.random.uniform(size=neg_batch_size)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        samples = np.concatenate((pos_samples, neg_samples))

        # binary labels indicating positive and negative samples
        labels = np.zeros(batch_size * (self.k + 1), dtype=np.float32)
        labels[:batch_size] = 1

        return th.from_numpy(samples), th.from_numpy(labels)

class SubgraphIterator:
    def __init__(self, g, num_rels, pos_sampler, sample_size=30000, num_epochs=6000):
        self.g = g
        self.num_rels = num_rels
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        if pos_sampler == 'neighbor':
            self.pos_sampler = NeighborExpand(g, sample_size)
        else:
            self.pos_sampler = GlobalUniform(g, sample_size)
        self.neg_sampler = NegativeSampler()

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, i):
        eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        # relabel nodes to have consecutive node IDs
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)
        # edges is the concatenation of src, dst with relabeled ID
        src, dst = np.reshape(edges, (2, -1))
        relabeled_data = np.stack((src, rel, dst)).transpose()

        samples, labels = self.neg_sampler.sample(relabeled_data, num_nodes)

        # Use only half of the positive edges
        chosen_ids = np.random.choice(np.arange(self.sample_size),
                                      size=int(self.sample_size / 2),
                                      replace=False)
        src = src[chosen_ids]
        dst = dst[chosen_ids]
        rel = rel[chosen_ids]
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_rels))
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = th.from_numpy(rel)
        sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = th.from_numpy(uniq_v).view(-1).long()

        return sub_g, uniq_v, samples, labels

# Utility functions for evaluations (raw)

def perturb_and_get_raw_rank(emb, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets"""
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    emb = emb.transpose(0, 1) # size D x V
    w = w.transpose(0, 1)     # size D x R
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = (idx + 1) * batch_size
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = emb[:,batch_a] * w[:,batch_r] # size D x E
        emb_ar = emb_ar.unsqueeze(2)           # size D x E x 1
        emb_c = emb.unsqueeze(1)               # size D x 1 x V

        # out-prod and reduce sum
        out_prod = th.bmm(emb_ar, emb_c)          # size D x E x V
        score = th.sum(out_prod, dim=0).sigmoid() # size E x V
        target = b[batch_start: batch_end]

        _, indices = th.sort(score, dim=1, descending=True)
        indices = th.nonzero(indices == target.view(-1, 1), as_tuple=False)
        ranks.append(indices[:, 1].view(-1))
    return th.cat(ranks)

# Utility functions for evaluations (filtered)

def filter(triplets_to_filter, target_s, target_r, target_o, num_nodes, filter_o=True):
    """Get candidate heads or tails to score"""
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)

    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]

    for e in range(num_nodes):
        triplet = (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter:
            candidate_nodes.append(e)
    return th.LongTensor(candidate_nodes)

def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter, filter_o=True):
    """Perturb subject or object in the triplets"""
    num_nodes = emb.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(triplets_to_filter, target_s, target_r,
                                 target_o, num_nodes, filter_o=filter_o)
        if filter_o:
            emb_s = emb[target_s]
            emb_o = emb[candidate_nodes]
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = 0
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = th.sigmoid(th.sum(emb_triplet, dim=1))

        _, indices = th.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return th.LongTensor(ranks)

def _calc_mrr(emb, w, test_mask, triplets_to_filter, batch_size, filter=False):
    with th.no_grad():
        test_triplets = triplets_to_filter[test_mask]
        s, r, o = test_triplets[:,0], test_triplets[:,1], test_triplets[:,2]
        test_size = len(s)

        if filter:
            metric_name = 'MRR (filtered)'
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter.tolist()}
            ranks_s = perturb_and_get_filtered_rank(emb, w, s, r, o, test_size,
                                                    triplets_to_filter, filter_o=False)
            ranks_o = perturb_and_get_filtered_rank(emb, w, s, r, o,
                                                    test_size, triplets_to_filter)
        else:
            metric_name = 'MRR (raw)'
            ranks_s = perturb_and_get_raw_rank(emb, w, o, r, s, test_size, batch_size)
            ranks_o = perturb_and_get_raw_rank(emb, w, s, r, o, test_size, batch_size)

        ranks = th.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed
        mrr = th.mean(1.0 / ranks.float()).item()
        print("{}: {:.6f}".format(metric_name, mrr))

    return mrr

# Main evaluation function

def calc_mrr(emb, w, test_mask, triplets, batch_size=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = _calc_mrr(emb, w, test_mask, triplets, batch_size, filter=True)
    else:
        mrr = _calc_mrr(emb, w, test_mask, triplets, batch_size)
    return mrr
