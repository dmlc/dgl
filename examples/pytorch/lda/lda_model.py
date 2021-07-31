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


def _edge_update(edges, mult=1):
    """ the Gibbs posterior distribution of z propto theta*beta.
    """
    q = edges.src['weight'] * edges.dst['weight']
    marg = q.sum(axis=1, keepdims=True) + np.finfo(float).eps
    p = q / marg

    return {
        'z': p * mult,
        'edge_elbo': marg.squeeze(1).log() * mult,
    }


def _bayesian_weight(z, prior, axis=1):
    """The sum is less than or equal to one according to Jensen's inequality:
    exp(E(log(x))) <= E(x) = 1. The bound is tight when z -> inf.
    """
    sum = z.shape[axis] * prior + z.sum(axis, keepdims=True)
    Elog = torch.digamma(prior + z) - torch.digamma(sum)
    return Elog.exp()


class LatentDirichletAllocation:
    """LDA model that works with a HeteroGraph with doc->word meta paths.
    The model alters the attributes of G arbitrarily.
    This is inspired by [1] and its corresponding scikit-learn implementation.

    Inputs
    ---
    * G: a template graph or an integer showing n_words
    * n_components: latent feature dimension; automatically set priors if missing.
    * prior: parameters in the Dirichlet prior; default to 1/n_components
    * lr: new_z = (1-lr)*old_z + lr*z; default to 1 for full gradients.
    * mult: multiplier for z-update; a large value effectively disables prior.
    * device: accelerate _bayesian_weight(word_z) during initialization.

    Caveat
    ---
    * With a larger n_components, the prior can get too small and _bayesian_weight
        outputs zeros. Suggest manually setting `prior={'doc': 0.1, 'word': 0.1}`.

    References
    ---
    [1] Matthew Hoffman, Francis Bach, David Blei. Online Learning for Latent
    Dirichlet Allocation. Advances in Neural Information Processing Systems 23
    (NIPS 2010).
    [2] Reactive LDA Library blogpost by Yingjie Miao for a similar Gibbs model
    """
    def __init__(
        self, G, n_components,
        prior=None,
        lr=1,
        mult={'doc': 1, 'word': 1},
        device='cpu',
        verbose=True,
        ):
        self.n_words = G.num_nodes('word') if hasattr(G, 'num_nodes') else G
        self.n_components = n_components

        if prior is None:
            prior = {'doc': 1./n_components, 'word': 1./n_components}
        self.prior = prior

        self.lr = lr
        self.mult = mult
        self.device = device
        self.verbose = verbose

        # Taken from scikit-learn.  Worked better than uniform.
        # The sample points concentrate around 1.0
        self._init = torch.distributions.gamma.Gamma(
            torch.tensor(100.0, device=device),
            torch.tensor(100.0, device=device),).sample

        self.word_z = self._init((self.n_words, n_components))
        self._word_weight = _bayesian_weight(self.word_z, self.prior['word'], axis=0)


    def _get_word_ids(self, G):
        if '_ID' in G.nodes['word'].data:
            return G.nodes['word'].data['_ID'].to(self.device)
        else:
            return slice(None)


    def _prepare_graph(self, G, doc_data=None):
        """ load or init node data; rewrite G.ndata inplace """
        if doc_data is None:
            z = self._init((G.num_nodes('doc'), self.n_components)).to(G.device)
            weight = _bayesian_weight(z, self.prior['doc'])
            doc_data = {'z': z, 'weight': weight}

        G.nodes['doc'].data['z'] = doc_data['z']
        G.nodes['doc'].data['weight'] = doc_data['weight']

        word_ids = self._get_word_ids(G)
        G.nodes['word'].data['z'] = self.word_z[word_ids].to(G.device)
        G.nodes['word'].data['weight'] = self._word_weight[word_ids].to(G.device)
        return G


    def _e_step(self, G, doc_data=None, mean_change_tol=1e-3, max_iters=100):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        G_d2w, G = G, self._prepare_graph(G.reverse(), doc_data) # word -> doc
        old_z = G.nodes['doc'].data['z']
        edge_fn = functools.partial(_edge_update, mult=self.mult['doc'])

        for i in range(max_iters):
            G.update_all(edge_fn, fn.sum('z', 'z'))
            doc_data = G.nodes['doc'].data
            doc_data['weight'] = _bayesian_weight(doc_data['z'], self.prior['doc'])

            mean_change = (doc_data['z'] - old_z).abs().mean()
            if mean_change < mean_change_tol:
                break
            old_z = doc_data['z']

        if self.verbose:
            print(f"e-step num_iters={i+1} with mean_change={mean_change:.4f}, "
                  f"perplexity={self.perplexity(G_d2w, doc_data):.4f}")

        return doc_data


    transform = _e_step


    def sample(self, doc_data, num_samples):
        doc_param = doc_data['z'] + self.prior['doc']
        word_param = (self.word_z + self.prior['word']).to(doc_param.device)

        z = torch.multinomial(doc_param, num_samples, True)
        w = torch.multinomial(word_param.T, num_samples, True)
        return torch.gather(w, 0, z)


    def _m_step(self, G, doc_data):
        """_m_step implements word data sampling and stores word_z stats.
        mean_change is in the sense of full graph with lr=1.
        """
        G = self._prepare_graph(G.clone(), doc_data)
        word_ids = self._get_word_ids(G)
        old_z = G.nodes['word'].data['z'].to(self.device)

        edge_fn = functools.partial(_edge_update, mult=self.mult['word'])
        G.update_all(edge_fn, fn.sum('z', 'z'))
        new_z = G.nodes['word'].data['z'].to(self.device)

        self._last_mean_change = (new_z - old_z).abs().mean().tolist()
        del old_z
        if self.verbose:
            print(f"m-step mean_change={self._last_mean_change:.4f}, ", end="")

        self.word_z *= (1 - self.lr)
        self.word_z[word_ids] += self.lr * new_z.to(self.device)
        del new_z

        self._word_weight = _bayesian_weight(self.word_z, self.prior['word'], axis=0)

        if self.verbose:
            print(f"weight_gap={1 - self._word_weight.sum(axis=0).mean():.4f}")


    def partial_fit(self, G):
        doc_data = self._e_step(G)
        self._m_step(G, doc_data)
        return self


    def fit(self, G, mean_change_tol=1e-3, max_epochs=10):
        for i in range(max_epochs):
            if self.verbose:
                print(f"epoch {i+1}, ", end="")
            self.partial_fit(G)

            if self._last_mean_change < mean_change_tol:
                break
        return self


    def _edge_elbo(self, G, doc_data):
        G = self._prepare_graph(G.reverse(), doc_data) # word -> doc
        edge_fn = functools.partial(_edge_update, mult=self.mult['doc'])
        G.update_all(edge_fn, fn.sum('edge_elbo', 'edge_elbo'))
        ndata = G.nodes['doc'].data
        return (ndata['edge_elbo'].sum() / ndata['z'].sum())


    def _node_elbo(self, ndata, ntype):
        """ Eq (4) in Hoffman et al., 2010.
        """
        axis = 1 if ntype=='doc' else 0
        z = ndata['z']
        weight = ndata['weight']
        prior = torch.tensor(self.prior[ntype], device=z.device, dtype=z.dtype)
        K = z.shape[axis]

        log_evid = (z * weight.log()).sum(axis)
        log_B_a = (K * torch.lgamma(prior) - torch.lgamma(K * prior))
        log_B_g = _lbeta(prior + z, axis)
        return (- log_evid - log_B_a + log_B_g).sum() / z.sum()


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

        ppl = np.exp(-edge_elbo - doc_elbo - word_elbo)
        if G.num_edges()>0 and np.isnan(ppl):
            warnings.warn("numerical issue in perplexity")
        return ppl


if __name__ == '__main__':
    print('Testing LatentDirichletAllocation ...')
    G = dgl.heterograph({('doc', '', 'word'): [(0, 0), (0, 3)]})
    model = LatentDirichletAllocation(G, 5, verbose=False)
    model.fit(G)
    model.transform(G)
    model.sample(model.transform(G), 2)
    model.perplexity(G)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
        G.reverse(), {'doc': np.arange(G.num_nodes('doc'))}, sampler,
        batch_size=1024, shuffle=True, drop_last=False)
    for input_nodes, _, (block,) in dataloader:
        B = dgl.DGLHeteroGraph(
            block._graph, ['_', 'word', 'doc', '_'], block.etypes
        ).reverse()
        B.nodes['word'].data.update(block.nodes['word'].data)
        model.partial_fit(B)
    print('Testing LatentDirichletAllocation passed!')
