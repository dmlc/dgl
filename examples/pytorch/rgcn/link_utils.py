"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhood expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        # Sample nodes already visited if possible
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        # Get adj_list of the sampled node
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        # Sample an edge connected to the sampled node
        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        # Sample without replacement
        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_uniform(n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def sample_subgraph(triplets, num_rels, adj_list, degrees,
                    sampler, sample_size, split_size, negative_rate):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    # edges is the concatenation of src, dst with relabeled ID
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.num_nodes())).float()
    norm = 1.0 / in_deg
    norm[torch.isinf(norm)] = 0
    return norm.unsqueeze(-1)

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node in-degree)
    """
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), norm.long()

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    # binary labels indicating positive and negative samples
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility functions for evaluations (raw)
#
#######################################################################

def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V

        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]

        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
        ranks.append(indices[:, 1].view(-1))
    return torch.cat(ranks)

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter(triplets_to_filter, target_s, target_r, target_o, num_entities, filter_o=True):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_entity = []
    for e in range(num_entities):
        # Do not consider an entity if it is part of a triplet to filter
        triplet = (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        # Do not filter out a test triplet, since we want to predict on it
        if triplet not in triplets_to_filter or triplet == (target_s, target_r, target_o):
            filtered_entity.append(e)
    return torch.LongTensor(filtered_entity)

def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter, filter_o=True):
    """Perturb subject or object in the triplets"""
    num_entities = emb.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_e = filter(triplets_to_filter, target_s, target_r,
                            target_o, num_entities, filter_o=filter_o)
        if filter_o:
            target_idx = int((filtered_e == target_o).nonzero())
            emb_s = emb[target_s]
            emb_o = emb[filtered_e]
        else:
            target_idx = int((filtered_e == target_s).nonzero())
            emb_s = emb[filtered_e]
            emb_o = emb[target_o]
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_mrr(emb, w, train_triplets, valid_triplets, test_triplets, batch_size, filter=False):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        if filter:
            metric_name = 'MRR (filtered)'
            triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets])
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter.tolist()}
            ranks_s = perturb_and_get_filtered_rank(emb, w, s, r, o, test_size,
                                                    triplets_to_filter, filter_o=False)
            ranks_o = perturb_and_get_filtered_rank(emb, w, s, r, o,
                                                    test_size, triplets_to_filter)
        else:
            metric_name = 'MRR (raw)'
            ranks_s = perturb_and_get_raw_rank(emb, w, o, r, s, test_size, batch_size)
            ranks_o = perturb_and_get_raw_rank(emb, w, s, r, o, test_size, batch_size)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed
        mrr = torch.mean(1.0 / ranks.float()).item()
        print("{}: {:.6f}".format(metric_name, mrr))

    return mrr

#######################################################################
#
# Main evaluation function
#
#######################################################################

def calc_mrr(embedding, w, train_triplets, valid_triplets,
             test_triplets, batch_size=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = calc_mrr(embedding, w, train_triplets, valid_triplets,
                       test_triplets, batch_size, filter=True)
    else:
        mrr = calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, batch_size)
    return mrr
