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


def _bayesian_weight(z, ntype, prior, lr_mult):
    """The sum is less than or equal to one according to Jensen's inequality:
    exp(E(log(x))) <= E(x) = 1. The bound is tight when z -> inf.
    """
    axis = 1 if ntype == 'doc' else 0 # word
    K = z.shape[axis]

    if lr_mult < float("inf"):
        sum = K * prior + z.sum(axis, keepdims=True) * lr_mult
        Elog = torch.digamma(prior + z * lr_mult) - torch.digamma(sum)
        return Elog.exp()
    else:
        return z / z.sum(axis, keepdims=True)


class LatentDirichletAllocation:
    """LDA model that works with a HeteroGraph with doc->word meta paths.
    The model alters the attributes of G arbitrarily.
    This is inspired by [1] and its corresponding scikit-learn implementation.

    Hyperparameters
    ---
    * prior: parameters in the Dirichlet prior; default to 1/n_components
    * lr: new_z = (1-lr)*old_z + lr*z; default to 1 for full gradients.
    * lr_mult: multiplier for z-update; use float("inf") to wash out the prior.
    * device: accelerate _bayesian_weight(word_z) during initialization.

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
        self._word_weight = self._bayesian_weight(self.word_z, 'word')


    def _bayesian_weight(self, z, ntype):
        return _bayesian_weight(z, ntype, self.prior[ntype], self.lr_mult[ntype])


    def _prepare_graph(self, G, doc_data=None):
        """ load or init node data; rewrite G.ndata inplace """
        if doc_data is None:
            z = self._init((G.num_nodes('doc'), self.n_components)).to(G.device)
            weight = self._bayesian_weight(z, 'doc')
            doc_data = {'z': z, 'weight': weight}

        G.nodes['doc'].data['z'] = doc_data['z']
        G.nodes['doc'].data['weight'] = doc_data['weight']

        G.nodes['word'].data['z'] = self.word_z.to(G.device)
        G.nodes['word'].data['weight'] = self._word_weight.to(G.device)
        return G


    def _e_step(self, G, doc_data=None, max_iters=100, mean_change_tol=1e-3):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        G = self._prepare_graph(G.reverse(), doc_data) # word -> doc

        for i in range(max_iters):
            old_z = G.nodes['doc'].data['z']
            G.update_all(_edge_update, fn.sum('z', 'z'))
            new_z = G.nodes['doc'].data['z']

            G.nodes['doc'].data['weight'] = self._bayesian_weight(new_z, 'doc')

            mean_change = (new_z - old_z).abs().mean()
            if mean_change < mean_change_tol:
                break

        if self.verbose:
            print(f'e-step num_iters={i+1} with mean_change={mean_change:.4f}')

        return dict(G.nodes['doc'].data)


    transform = _e_step


    def _m_step(self, G, doc_data):
        """_m_step implements word data sampling and stores word_z stats
        """
        G = self._prepare_graph(G.clone(), doc_data)

        old_z = G.nodes['word'].data['z'].to(self.device)
        G.update_all(_edge_update, fn.sum('z', 'z'))
        new_z = G.nodes['word'].data['z'].to(self.device)

        mean_change = (new_z - old_z).abs().mean()
        del old_z

        self.word_z = (1 - self.lr) * self.word_z + self.lr * new_z
        del new_z

        self._word_weight = self._bayesian_weight(self.word_z, 'word')
        return mean_change


    def fit(self, G, batch_size=0, max_iters=10, mean_change_tol=1e-3):
        if batch_size>0:
            raise NotImplementedError("""
                Use a custom loop to iterate between e and m steps;
                set self.lr<1 and follow the example at the end of this file.""")

        for i in range(max_iters):
            doc_data = self._e_step(G)
            mean_change = self._m_step(G, doc_data)

            if self.verbose:
                print(f"iter {i+1}, m-step mean_change={mean_change:.4f}, "
                      f"digamma_gap={1 - self._word_weight.sum(axis=0).mean():.4f}, "
                      f"perplexity={self.perplexity(G, doc_data):.4f}")

            if mean_change < mean_change_tol:
                break
        return self


    def _edge_elbo(self, G, doc_data):
        G = self._prepare_graph(G.reverse(), doc_data) # word -> doc
        G.update_all(_edge_update, fn.sum('edge_elbo', 'edge_elbo'))
        ndata = G.nodes['doc'].data
        return (ndata['edge_elbo'].sum() / ndata['z'].sum())


    def _node_elbo(self, ndata, ntype):
        """ Eq (4) in Hoffman et al., 2010.
        """
        lr_mult = self.lr_mult[ntype]
        z = ndata['z']
        weight = ndata['weight']
        prior = torch.tensor(self.prior[ntype], device=z.device)

        axis = 1 if ntype=='doc' else 0 # word
        K = z.shape[axis]

        if lr_mult < float("inf"):
            return (
                (-z * weight.log()).sum(axis) * lr_mult
                - (K * torch.lgamma(prior) - torch.lgamma(K * prior)) # logB(a)
                + _lbeta(prior + z * lr_mult, axis)               # logB(gamma)
            ).sum() / z.sum() / lr_mult
        else:
            return torch.tensor(0.0, device=z.device)


    def perplexity(self, G, doc_data=None):
        """ppl = exp{-sum[log(p(w1,...,wn|d))] / n}
        Follows Eq (15) in Hoffman et al., 2010.
        """
        if doc_data is None:
            doc_data = self._e_step(G)
        word_data = {'z': self.word_z, 'weight': self._word_weight}

        # compute E[log p(docs | theta, beta)]
        edge_elbo = self._edge_elbo(G, doc_data).cpu().numpy()
        if self.verbose:
            print(f'neg_elbo phi: {-edge_elbo:.3f}', end=' ')

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        doc_elbo = self._node_elbo(doc_data, 'doc').cpu().numpy()
        if self.verbose:
            print(f'theta: {-doc_elbo:.3f}', end=' ')

        # compute E[log p(beta | eta) - log q(beta | lambda)]
        # The denominator z.sum() for extrapolation perplexity is undefined.
        # We use the train set, whereas sklearn uses the test set.
        word_elbo = self._node_elbo(word_data, 'word').cpu().numpy()
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
    print('Testing LatentDirichletAllocation passed!')
