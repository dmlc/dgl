"""
By Minjie
"""


from __future__ import division

import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

class SSBM:
    def __init__(self, n, k, a=10.0, b=2.0, regime='constant', rng=None):
        """Symmetric Stochastic Block Model.
         n - number of nodes
         k - number of communities
         a - probability scale for intra-community edge
         b - probability scale for inter-community edge
         regime - If "logaritm", this generates SSBM(n, k, a*log(n)/n, b*log(n)/n)
                  If "constant", this generates SSBM(n, k, a/n, b/n)
                  If "mixed", this generates SSBM(n, k, a*log(n)/n, b/n)
        """
        self.n = n
        self.k = k
        if regime == 'logarithm':
            if math.sqrt(a) - math.sqrt(b) >= math.sqrt(k):
                print('SSBM model with possible exact recovery.')
            else:
                print('SSBM model with impossible exact recovery.')
            self.a = a * math.log(n) / n
            self.b = b * math.log(n) / n
        elif regime == 'constant':
            snr = (a - b) ** 2 / (k * (a + (k - 1) * b))
            if snr > 1:
                print('SSBM model with possible detection.')
            else:
                print('SSBM model that may not have detection (snr=%.5f).' % snr)
            self.a = a / n
            self.b = b / n
        elif regime == 'mixed':
            self.a = a * math.log(n) / n
            self.b = b / n
        else:
            raise ValueError('Unknown regime: %s' % regime)
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        self._graph = None

    def generate(self):
        self.generate_communities()
        print('Finished generating communities.')
        self.generate_edges()
        print('Finished generating edges.')

    def generate_communities(self):
        nodes = list(range(self.n))
        size = self.n // self.k
        self.block_size = size
        self.comm2node = [nodes[i*size:(i+1)*size] for i in range(self.k)]
        self.node2comm = [nid // size for nid in range(self.n)]

    def generate_edges(self):
        # TODO: dedup edges
        us = []
        vs = []
        # generate intra-comm edges
        for i in range(self.k):
            sp_mat = sp.random(self.block_size, self.block_size,
                               density=self.a,
                               random_state=self.rng,
                               data_rvs=lambda l: np.ones(l))
            u = sp_mat.row + i * self.block_size
            v = sp_mat.col + i * self.block_size
            us.append(u)
            vs.append(v)
        # generate inter-comm edges
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    continue
                sp_mat = sp.random(self.block_size, self.block_size,
                                   density=self.b,
                                   random_state=self.rng,
                                   data_rvs=lambda l: np.ones(l))
                u = sp_mat.row + i * self.block_size
                v = sp_mat.col + j * self.block_size
                us.append(u)
                vs.append(v)
        us = np.hstack(us)
        vs = np.hstack(vs)
        self.sp_mat = sp.coo_matrix((np.ones(us.shape[0]), (us, vs)), shape=(self.n, self.n))

    @property
    def graph(self):
        if self._graph is None:
            self._graph = nx.from_scipy_sparse_matrix(self.sp_mat, create_using=nx.DiGraph())
        return self._graph

    def plot(self):
        x = self.sp_mat.row
        y = self.sp_mat.col
        plt.scatter(x, y, s=0.5, marker='.', c='k')
        plt.savefig('ssbm-%d-%d.pdf' % (self.n, self.k))
        plt.clf()
        # plot out degree distribution
        out_degree = [d for _, d in self.graph.out_degree().items()]
        plt.hist(out_degree, 100, normed=True)
        plt.savefig('ssbm-%d-%d_out_degree.pdf' % (self.n, self.k))
        plt.clf()

if __name__ == '__main__':
    n = 1000
    k = 10
    ssbm = SSBM(n, k, regime='mixed', a=4, b=1)
    ssbm.generate()
    g = ssbm.graph
    print('#nodes:', g.number_of_nodes())
    print('#edges:', g.number_of_edges())
    #ssbm.plot()
    #lg = nx.line_graph(g)
    # plot degree distribution
    #degree = [d for _, d in lg.degree().items()]
    #plt.hist(degree, 100, normed=True)
    #plt.savefig('lg<ssbm-%d-%d>_degree.pdf' % (n, k))
    #plt.clf()
