import ..backend as F
from .. import convert
from .. import function as fn

import numpy as np

class PinSAGESampler(object):
    def __init__(self, G, ntype, other_type, random_walk_length, random_walk_restart_prob):
        self.G = G

        metagraph = G.metagraph
        fw_etype = list(g[ntype][other_type])[0]
        bw_etype = list(g[other_type][ntype])[0]
        self.metapath = [fw_etype, bw_etype] * random_walk_length
        restart_prob = np.tile(np.array([0, random_walk_restart_prob]), random_walk_length)
        self.restart_prob = F.zerocopy_from_numpy(restart_prob)

    def generate(self, seed_nodes):
        paths = random_walk(
                self.G, seed_nodes, metapath=self.metapath, restart_prob=self.restart_prob)
        src = F.reshape(paths[:, 2::2], (-1,))
        dst = F.reshape(F.repeat(paths[:, 0:1], F.shape(dst)[1], 1), (-1,))

        src_mask = (src != -1)
        src = F.boolean_mask(src, src_mask)
        dst = F.boolean_mask(dst, src_mask)

        neighbor_graph = convert.graph((src, dst))
        neighbor_graph = convert.to_simple(neighbor_graph, weights='weights')
        neighbor_graph = convert.select_topk(neighbor_graph, weights='weights')

        neighbor_graph.update_all(fn.copy_e('weights', 'm'), fn.sum('m', 'total_weights'))
        neighbor_graph.apply_edges(
            lambda edges: {'weights': edges.data['weights'] / edges.dst['total_weights']})

        return neighbor_graph
