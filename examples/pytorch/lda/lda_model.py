# Copyright 2021 Yifei Ma
# with references from "sklearn.decomposition.LatentDirichletAllocation"
# with the following original authors:
# * Chyi-Kwei Yau (the said scikit-learn implementation)
# * Matthew D. Hoffman (original onlineldavb implementation)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os, functools, warnings
import numpy as np, scipy as sp
import torch
import dgl
from dgl import function as fn


def _lbeta(alpha, axis):
    return torch.lgamma(alpha).sum(axis) - torch.lgamma(alpha.sum(axis))


def _edge_update(edges):
    """ the Gibbs posterior distribution of z propto theta*beta.
    """
    q = edges.src['weight'] * edges.dst['weight']
    marg = q.sum(axis=1, keepdims=True) + np.finfo(float).eps
    p = q / marg

    return {
        'z': p,
        'edge_elbo': marg.squeeze(1).log(),
    }


def _bayesian_softmax(z, ntype, prior):
    """The sum is less than or equal to one according to Jensen's inequality:
    exp(E(log(x))) <= E(x) = 1. The bound is tight when z is large, e.g., due
    to large lr_mult in the corresponding direction.
    """
    gamma = prior + z

    axis = 1 if ntype == 'doc' else 0 # word
    Elog = torch.digamma(gamma) - torch.digamma(gamma.sum(axis, keepdims=True))

    return Elog.exp()


class LatentDirichletAllocation:
    """LDA model that works with a HeteroGraph with doc->word meta paths.
    The model alters the attributes of G arbitrarily, starting with word->weight.
    This is inspired by [1] and its corresponding scikit-learn implementation.

    Hyperparameters
    ---
    * prior: parameters in the Dirichlet prior; default to 1/n_components
    * lr: learning rate for online update; default to 1 for full gradient updates
    * lr_mult: learning rate multiplier to quickly converge to MAP
    * device: accelerate _bayesian_softmax on word_z parameters in m-step
    * Tolerance / max_iters parameters are less often accessed; monkey-patch if needed

    References
    ---
    [1] Matthew Hoffman, Francis Bach, David Blei. Online Learning for Latent
    Dirichlet Allocation. Advances in Neural Information Processing Systems 23
    (NIPS 2010).
    [2] Reactive LDA Library blogpost by Yingjie Miao for a similar Gibbs model
    """
    def __init__(
        self, n_words, n_components,
        prior=None,
        lr=1,
        lr_mult={'doc': 1, 'word': 1},
        device='cpu',
        verbose=True,
        _e_step_max_iters=100,
        _e_step_mean_change_tol=1e-3,
        _m_step_max_iters=10,
        _m_step_mean_change_tol=1e-3,
        ):
        self.n_components = n_components

        if prior is None:
            prior = {'doc': 1./n_components, 'word': 1./n_components}
        self.prior = prior

        self.lr = lr
        self.lr_mult = lr_mult
        self.device = device
        self.verbose = verbose

        # Taken from scikit-learn.  Worked better than uniform.
        # The sample points concentrate around 1.0
        self._init = torch.distributions.gamma.Gamma(
            torch.tensor(100.0, device=device),
            torch.tensor(100.0, device=device),).sample

        self.word_z = self._init((n_words, self.n_components))
        self._word_weight = _bayesian_softmax(self.word_z, 'word', self.prior['word'])

        self._e_step_max_iters=_e_step_max_iters
        self._e_step_mean_change_tol=_e_step_mean_change_tol
        self._m_step_max_iters=_m_step_max_iters
        self._m_step_mean_change_tol=_m_step_mean_change_tol


    def _prepare_graph(self, G, doc_data=None):
        """ asssume full set of iid docs and allow subset of word_ids """
        if doc_data is None:
            z = self._init((G.num_nodes('doc'), self.n_components)).to(G.device)
            weight = _bayesian_softmax(z, 'doc', self.prior['doc'])
            doc_data = {'z': z, 'weight': weight}

        G.nodes['doc'].data['z'] = doc_data['z']
        G.nodes['doc'].data['weight'] = doc_data['weight']

        # word_ids = G.in_degrees().nonzero(as_tuple=True)[0]

        if 'word_ids' in G.nodes['word'].data:
            word_ids = G.nodes['word'].data['word_ids'].to(self.device)
        else:
            word_ids = slice(None)

        G.nodes['word'].data['z'] = self.word_z[word_ids].to(G.device)
        G.nodes['word'].data['weight'] = self._word_weight[word_ids].to(G.device)

        return word_ids


    def _e_step(self, G, doc_data=None):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        G = G.reverse() # word -> doc
        self._prepare_graph(G, doc_data)

        for i in range(self._e_step_max_iters):
            doc_z_old = G.nodes['doc'].data['z']

            G.update_all(
                _edge_update, fn.sum('z', 'z'), 
                lambda x: {'z': x.data['z'] * self.lr_mult['doc']}
            )
            G.nodes['doc'].data['weight'] = _bayesian_softmax(
                G.nodes['doc'].data['z'], 'doc', self.prior['doc'])

            mean_change = (G.nodes['doc'].data['z'] - doc_z_old).abs().mean()
            if mean_change < self._e_step_mean_change_tol:
                break

        if self.verbose:
            print(f'e-step num_iters={i+1} with mean_change={mean_change:.4f}')

        return dict(G.nodes['doc'].data)


    transform = _e_step


    def _m_step(self, G, doc_data):
        """_m_step implements word data sampling and stores word_z stats
        """
        word_ids = self._prepare_graph(G, doc_data)

        G.update_all(
            _edge_update, fn.sum('z', 'z'),
            lambda x: {'z': x.data['z'] * self.lr_mult['word']}
        )
        new = G.nodes['word'].data['z'].to(self.device)
        word_z_diff = new - self.word_z[word_ids]

        self.word_z = self.word_z * (1 - self.lr)
        self.word_z[word_ids] += self.lr * new     # zero everywhere else
        # softmax on the full word distribution
        self._word_weight = _bayesian_softmax(self.word_z, 'word', self.prior['word'])

        return word_z_diff


    def fit(self, G, batch_size=0):
        if batch_size>0:
            raise NotImplementedError("""
                Use a custom loop to iterate between e and m steps;
                set self.lr<1 and follow the example at the end of this file.""")

        for i in range(self._m_step_max_iters):
            doc_data = self._e_step(G)
            mean_change = self._m_step(G, doc_data).abs().mean()

            if self.verbose:
                print(f"iter {i+1}, m-step mean_change: {mean_change:.4f}, "
                      f"perplexity: {self.perplexity(G, doc_data):.4f}")

            if mean_change < self._m_step_mean_change_tol:
                break
        return self


    def perplexity(self, G, doc_data=None):
        """ppl = exp{-sum[log(p(w1,...,wn|d))] / n}
        Follows Eq (15) in Hoffman et al., 2010.
        """
        if doc_data is None:
            doc_data = self._e_step(G)
        word_data = {'z': self.word_z, 'weight': self._word_weight}

        # augment doc_data with edge_elbo
        G = G.reverse()
        self._prepare_graph(G, doc_data)
        G.update_all(
            _edge_update, fn.sum('edge_elbo', 'edge_elbo'),
            lambda x: {"edge_elbo": x.data['edge_elbo'] * self.lr_mult['doc']}
        )
        doc_data = dict(G.nodes['doc'].data)

        # compute E[log p(docs | theta, beta)]
        edge_elbo = (
            doc_data['edge_elbo'].sum() / doc_data['z'].sum()
        ).cpu().numpy()
        if self.verbose:
            print(f'neg_elbo phi: {-edge_elbo:.3f}', end=' ')

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        doc_elbo = (
            (-doc_data['z'] * doc_data['weight'].log()).sum(axis=1)
            -_lbeta(self.prior['doc'] + doc_data['z'] * 0, axis=1)
            +_lbeta(self.prior['doc'] + doc_data['z'], axis=1)
        )
        doc_elbo = (doc_elbo.sum() / doc_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'theta: {-doc_elbo:.3f}', end=' ')

        # compute E[log p(beta | eta) - log q (beta | lambda)]
        word_elbo = (
            (-word_data['z'] * word_data['weight'].log()).sum(axis=0)
            -_lbeta(self.prior['word'] + word_data['z'] * 0, axis=0)
            +_lbeta(self.prior['word'] + word_data['z'], axis=0)
        )
        word_elbo = (word_elbo.sum() / word_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'beta: {-word_elbo:.3f}')

        return np.exp(-edge_elbo - doc_elbo - word_elbo)


if __name__ == '__main__':
    print('Testing LatentDirichletAllocation ...')
    G = dgl.heterograph({('doc', '', 'word'): [(0, 0), (0, 3)]})
    model = LatentDirichletAllocation(G.num_nodes('word'), 5, verbose=False)
    model.fit(G)
    model.transform(G)
    model.perplexity(G)

    dataloader = dgl.dataloading.NodeDataLoader(
        G.reverse(),         # sample by in-degree, to be reversed in blocks
        {'doc': np.arange(G.num_nodes('doc'))},
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
    )

    model = LatentDirichletAllocation(G.num_nodes('word'), 5, verbose=False, lr=0.1)
    for input_nodes, output_nodes, blocks in dataloader:
        word2doc = blocks[0].adjacency_matrix()._indices().T.tolist()
        B = dgl.heterograph({('word', '', 'doc'): word2doc}).reverse()
        B.nodes['word'].data['word_ids'] = input_nodes['word'].to(B.device)

        model._m_step(B, model._e_step(B))

    print('Testing LatentDirichletAllocation passed!')
