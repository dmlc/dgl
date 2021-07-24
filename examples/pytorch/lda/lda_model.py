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

# Taken from scikit-learn.  Worked better than uniform.
# Perhaps this is due to concentration around one.
_sklearn_random_init = torch.distributions.gamma.Gamma(100, 100).sample


def _edge_update(edges, step_size=1):
    """ the Gibbs posterior distribution of z propto theta*beta.
    As step_size -> infty, the result becomes MAP estimate on the dst nodes.
    """
    q = edges.src['weight'] * edges.dst['weight']
    marg = q.sum(axis=1, keepdims=True) + np.finfo(float).eps
    p = q / marg

    return {
        'z': p * step_size,
        'edge_elbo': marg.squeeze(1).log() * step_size,
    }


def _weight_fn(z, ntype, prior):
    """Node weight is approximately normalized for VB along the ntype
    direction.
    """
    prior = prior + z * 0 # convert numpy to torch
    gamma = prior + z

    axis = 1 if ntype == 'doc' else 0 # word
    Elog = torch.digamma(gamma) - torch.digamma(gamma.sum(axis, keepdims=True))

    return Elog.exp()


def _message_passing(G, dst_type, prior, step_size):
    """Follows Eq (5) of Hoffman et al., 2010.
    """
    G = G.reverse() if dst_type == 'doc' else G # word
    msg_fn = lambda edges: _edge_update(edges, step_size)

    G.update_all(msg_fn, fn.sum('z','z'))
    G.update_all(msg_fn, fn.sum('edge_elbo', 'edge_elbo'))

    out = dict(G.nodes[dst_type].data)
    out['weight'] = _weight_fn(out['z'], dst_type, prior)
    return out


def _load_z_update_weight(G, ntype, z, prior):
    z = z.to(G.device)
    G.nodes[ntype].data['z'] = z
    G.nodes[ntype].data['weight'] = _weight_fn(z, ntype, prior)
    return dict(G.nodes[ntype].data)


class LatentDirichletAllocation:
    """LDA model that works with a HeteroGraph with doc/word node types.
    The model alters the attributes of G arbitrarily,
    but always load word_z if needed.
    This is inspired by [1] and its corresponding scikit-learn implementation.

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
        step_size={'doc': 1, 'word': 1}, # use larger value to get MAP
        word_rho=1, # use smaller value for online update
        verbose=True,
        ):
        self.n_components = n_components

        if prior is None:
            prior = {'doc': 1./n_components, 'word': 1./n_components}
        self.prior = prior

        self.step_size = step_size

        self.word_rho = word_rho
        self.word_z = _sklearn_random_init((n_words, self.n_components))
        self.verbose = verbose

    def _e_step(self, G, mean_change_tol=1e-3, max_iters=100, word_ids=slice(None)):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        _load_z_update_weight(G, 'word', self.word_z[word_ids], self.prior['word'])
        doc_z = _sklearn_random_init((G.num_nodes('doc'), self.n_components)).to(G.device)

        for i in range(max_iters):
            _load_z_update_weight(G, 'doc', doc_z, self.prior['doc'])
            doc_data = _message_passing(
                G, 'doc', self.prior['doc'], self.step_size['doc'])

            mean_change = (doc_data['z'] - doc_z).abs().mean()
            doc_z = doc_data['z']
            if mean_change < mean_change_tol:
                break
        if self.verbose:
            print(f'e-step num_iters={i+1} with mean_change={mean_change:.4f}')
        return doc_data

    transform = _e_step

    def partial_fit(self, G, word_ids=slice(None)):
        """_m_step implements word data sampling and stores word_z stats
        """
        doc_data = self._e_step(G, word_ids=word_ids)
        word_data = _message_passing(
            G, 'word', self.prior['word'], self.step_size['word'])
        word_z = word_data['z'].to(self.word_z.device)

        self._last_mean_change = (word_z - self.word_z[word_ids]).abs().mean()
        self.word_z[word_ids] *= (1 - self.word_rho)
        self.word_z[word_ids] += self.word_rho * word_z

        if self.verbose:
            print(f'm-step mean_change: {self._last_mean_change:.4f}, '
                  f'perplexity: {self.perplexity(G, doc_data)}')
        return self

    def fit(self, G, mean_change_tol=1e-3, max_epochs=10, word_ids=slice(None)):
        for i in range(max_epochs):
            if self.verbose:
                print(f'epoch {i+1}, ', end='')
            self.partial_fit(G, word_ids=word_ids)
            if self._last_mean_change < mean_change_tol:
                break
        return self

    def perplexity(self, G, doc_data=None, word_ids=slice(None)):
        """ppl = exp{-sum[log(p(w1,...,wn|d))] / n}
        Follows Eq (15) in Hoffman et al., 2010.
        """
        word_data = _load_z_update_weight(
            G, 'word', self.word_z[word_ids], self.prior['word'])
        if doc_data is None:
            doc_data = self._e_step(G, word_ids=word_ids)

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
    print('Testing LatentDirichletAllocation via task_example_test.sh ...')
    n_words = 100
    G_train = dgl.heterograph(
        {('doc','topic','word'): [(0, 0), (0, 3)]},
        {'doc': 2, 'word': n_words}
    )
    G_test = dgl.heterograph(
        {('doc', 'topic', 'word'): [(0, 1), (1, 2)]},
        {'doc': 3, 'word': n_words}
    )
    model = LatentDirichletAllocation(n_words, 5, verbose=False)
    model.fit(G_train)
    model.transform(G_test)
    model.perplexity(G_test)
